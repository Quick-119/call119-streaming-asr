"""Experimental decoder batching with per-call text offsets and padding masks.

Calls with different context lengths can share a decoder forward pass. Saved
self-attention KV tensors are right-padded, and the new token is appended after
that padding. A per-call attention mask excludes only the padding; caches and
AlignAtt rows are unpadded before each call resumes.
"""
import torch
import torch.nn.functional as F
from greenlet import greenlet

from decoder_batching import DecoderBatcher


class VariableDecoderBatcher(DecoderBatcher):
    def __init__(self, engine, max_batch=20, bucket_width=8):
        super().__init__(engine, max_batch)
        if bucket_width < 1:
            raise ValueError("bucket_width must be positive")
        self.bucket_width = bucket_width
        self.variable_batches = 0
        self.padded_positions = 0

    def _signature(self, item):
        cache = item["state"]["kv_cache"]
        if not cache:
            length = item["tokens"].shape[1]
            return ("initial", (length - 1) // self.bucket_width)
        if item["tokens"].shape[1] == 1:
            self_keys = sorted(k for k in cache if "_self_attn_" in k)
            if not self_keys:
                raise RuntimeError("Incremental decode has no self-attention cache")
            length = cache[self_keys[0]].shape[1]
            return ("incremental", length // self.bucket_width)
        return super()._signature(item)

    @staticmethod
    def _masked_self_attention(module, invalid_cache_mask, valid_cache_mask):
        original = module.qkv_attention

        def masked(q, k, v, mask=None):
            n_batch, n_ctx, n_state = q.shape
            scale = (n_state // module.n_head) ** -0.25
            q = q.view(n_batch, n_ctx, module.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(k.shape[0], k.shape[1], module.n_head, -1).permute(0, 2, 1, 3)
            v = v.view(v.shape[0], v.shape[1], module.n_head, -1).permute(0, 2, 1, 3)
            if getattr(module, "encoder_only_sdpa", False):
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=valid_cache_mask)
                return output.permute(0, 2, 1, 3).flatten(start_dim=2), None
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float() + invalid_cache_mask
            weights = F.softmax(qk, dim=-1).to(q.dtype)
            output = (weights @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            return output, qk.detach()

        module.qkv_attention = masked
        return original

    def _decode_group(self, group):
        if any(item["tokens"].shape[1] == 0 for item in group):
            raise RuntimeError("Zero-length decoder tokens after variable batching: " +
                               repr({"calls": [item["cid"] for item in group],
                                     "token_shapes": [tuple(item["tokens"].shape) for item in group],
                                     "cache_items": [len(item["state"]["kv_cache"]) for item in group],
                                     "variable_batches": self.variable_batches}))
        if len(group) == 1 or len({super(VariableDecoderBatcher, self)._signature(x) for x in group}) == 1:
            return super()._decode_group(group)
        m = self.model
        decoder = m.model.decoder
        initial = not group[0]["state"]["kv_cache"]
        if any(bool(item["state"]["kv_cache"]) == initial for item in group):
            raise RuntimeError("Mixed initial and incremental decoder calls")
        token_lengths = [item["tokens"].shape[1] for item in group]
        if not initial and any(length != 1 for length in token_lengths):
            raise RuntimeError("Expected one incremental token per call")
        offsets = [] if initial else [next(value.shape[1] for key, value in item["state"]["kv_cache"].items()
                                               if "_self_attn_" in key) for item in group]
        max_input = max(token_lengths)
        max_offset = max(offsets) if offsets else 0
        self.padded_positions += sum(max_input - length for length in token_lengths)
        self.padded_positions += sum(max_offset - length for length in offsets)
        tokens = torch.cat([F.pad(item["tokens"], (0, max_input - item["tokens"].shape[1]))
                            for item in group], dim=0)
        features = torch.cat([item["features"].repeat_interleave(self.beam, dim=0)
                              for item in group], dim=0)
        cache = {}
        if not initial:
            keys = list(group[0]["state"]["kv_cache"])
            if not keys or "_self_attn_" not in keys[0]:
                raise RuntimeError("Decoder cache must start with a text self-attention key")
            if any(set(item["state"]["kv_cache"]) != set(keys) for item in group):
                raise RuntimeError("Decoder cache keys differ across calls")
            for key in keys:
                parts = []
                for item, offset in zip(group, offsets):
                    value = item["state"]["kv_cache"][key]
                    if "_cross_attn_" in key:
                        if value.shape[0] != 1:
                            raise RuntimeError("Expected one cached encoder sequence per call")
                        value = value.repeat_interleave(self.beam, dim=0)
                    else:
                        if value.shape[0] != self.beam or value.shape[1] != offset:
                            raise RuntimeError("Invalid self-attention cache shape")
                        value = F.pad(value, (0, 0, 0, max_offset - offset))
                    parts.append(value)
                cache[key] = torch.cat(parts, dim=0)
        m.kv_cache = cache
        if self.beam > 1:
            m.inference.kv_cache = cache
        m.dec_attns = []
        saved_attention = []
        try:
            if initial:
                logits = decoder(tokens, features, kv_cache=cache)
            else:
                invalid = torch.zeros((self.beam * len(group), 1, 1, max_offset + 1),
                                      device=tokens.device, dtype=torch.float32)
                for i, offset in enumerate(offsets):
                    invalid[self.beam * i:self.beam * (i + 1), :, :, offset:max_offset] = -torch.inf
                valid = invalid == 0
                for block in decoder.blocks:
                    module = block.attn
                    saved_attention.append((module, self._masked_self_attention(module, invalid, valid)))
                offset_tensor = torch.tensor(offsets, device=tokens.device).repeat_interleave(self.beam)
                x = decoder.token_embedding(tokens) + decoder.positional_embedding[offset_tensor, :].unsqueeze(1)
                for block in decoder.blocks:
                    x = block(x, features, mask=decoder.mask, kv_cache=cache)
                x = decoder.ln(x)
                logits = x @ decoder.token_embedding.weight.T
        finally:
            for module, original in saved_attention:
                module.qkv_attention = original
        self.batch_sizes.append(len(group))
        self.variable_batches += 1
        output = []
        for i, item in enumerate(group):
            low, high = self.beam * i, self.beam * (i + 1)
            token_length = token_lengths[i]
            state = item["state"].copy()
            new_cache = {}
            for key, value in m.kv_cache.items():
                if "_cross_attn_" in key:
                    new_cache[key] = value[low:low + 1]
                elif initial:
                    new_cache[key] = value[low:high, :token_length]
                else:
                    new_cache[key] = torch.cat((value[low:high, :offsets[i]],
                                                value[low:high, -1:]), dim=1)
            state["kv_cache"] = new_cache
            state["dec_attns"] = state["dec_attns"] + [
                x[low:high, :, :token_length if initial else 1, :].squeeze(0)
                if self.beam == 1 else
                x[low:high, :, :token_length if initial else 1, :]
                for x in m.dec_attns]
            output.append((item, logits[low:high, :token_length], state))
        return output

    def finish_many(self, call_ids, service_urgent=None):
        """Flush multiple active calls with the same cooperative decoder loop."""
        if not call_ids:
            return {}
        engine = self.engine
        with engine.lock, torch.inference_mode(), engine.compute_context():
            pending = {}
            results = {}

            def advance(cid, coroutine, value=None, first=False):
                self.active_call_id = cid
                message = coroutine.switch() if first else coroutine.switch(value)
                if message[0] == "done":
                    results[cid] = message[1]
                elif message[0] == "logits":
                    pending[cid] = {"cid": cid, "coroutine": coroutine,
                                    "tokens": message[1], "features": message[2],
                                    "state": self._capture()}
                else:
                    raise RuntimeError("Unexpected finish coroutine response")

            try:
                for cid in call_ids:
                    session = engine.sessions[cid]
                    def run(saved_session=session):
                        engine._restore(saved_session["model"])
                        result = saved_session["vac"].finish()
                        return ("done", result)
                    advance(cid, greenlet(run), first=True)
                while pending:
                    if service_urgent is not None:
                        service_urgent()
                    groups = {}
                    for item in pending.values():
                        groups.setdefault(self._signature(item), []).append(item)
                    pending = {}
                    decoded = []
                    for group in groups.values():
                        for i in range(0, len(group), self.max_batch):
                            decoded.extend(self._decode_group(group[i:i + self.max_batch]))
                    for item, logits, state in decoded:
                        self._restore(state)
                        advance(item["cid"], item["coroutine"], logits)
                for cid in call_ids:
                    del engine.sessions[cid]
                return results
            except Exception:
                for cid in call_ids:
                    engine.sessions.pop(cid, None)
                raise
            finally:
                self.active_call_id = None
