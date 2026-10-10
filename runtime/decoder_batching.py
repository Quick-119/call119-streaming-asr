"""Experimental cooperative batching of SimulStreaming decoder calls.

Greenlets pause independent VAC updates at logits(). The shared decoder then
processes compatible requests in one forward pass. Beam=1 and beam=3 are
supported. Each call retains its own model, beam, KV, and AlignAtt state.
"""
from collections import defaultdict
from greenlet import greenlet, getcurrent
import numpy as np
import torch

from shared_streaming import STATE_FIELDS


class DecoderBatcher:
    def __init__(self, engine, max_batch=20):
        if engine.cfg["beam"] not in (1, 3):
            raise ValueError("DecoderBatcher supports beam=1 or beam=3")
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.engine = engine
        self.max_batch = max_batch
        self.model = engine.asr.model
        self.beam = engine.cfg["beam"]
        self.original_logits = self.model.logits
        self.active_call_id = None
        self.model.logits = self._pause_logits
        self.batch_sizes = []

    def _pause_logits(self, tokens, audio_features):
        if self.active_call_id is None:
            raise RuntimeError("Decoder batching logits called outside push_many")
        return getcurrent().parent.switch(("logits", tokens, audio_features))

    def _capture(self):
        m = self.model
        state = {name: getattr(m, name) for name in STATE_FIELDS}
        state["dec_attns"] = m.dec_attns
        state["kv_cache"] = m.kv_cache
        if self.beam > 1:
            state["finished_sequences"] = m.token_decoder.finished_sequences
        return state

    def _restore(self, state):
        m = self.model
        for name in STATE_FIELDS:
            setattr(m, name, state[name])
        m.dec_attns = state["dec_attns"]
        m.kv_cache = state["kv_cache"]
        if self.beam > 1:
            m.inference.kv_cache = m.kv_cache
            m.token_decoder.finished_sequences = state["finished_sequences"]

    @staticmethod
    def _signature(item):
        cache = item["state"]["kv_cache"]
        return (item["tokens"].shape[1],
                tuple((key, value.shape[0], value.shape[1])
                      for key, value in sorted(cache.items())))

    def _decode_group(self, group):
        m = self.model
        if len(group) == 1:
            item = group[0]
            self._restore(item["state"])
            logits = self.original_logits(item["tokens"], item["features"])
            return [(item, logits, self._capture())]

        batch_tokens = torch.cat([item["tokens"] for item in group], dim=0)
        batch_features = torch.cat([item["features"].repeat_interleave(self.beam, dim=0)
                                    for item in group], dim=0)
        keys = group[0]["state"]["kv_cache"].keys()
        cache = {}
        for key in keys:
            pieces = []
            for item in group:
                value = item["state"]["kv_cache"][key]
                if "_cross_attn_" in key:
                    if value.shape[0] != 1:
                        raise RuntimeError("Expected one cached encoder sequence per call")
                    value = value.repeat_interleave(self.beam, dim=0)
                elif value.shape[0] != self.beam:
                    raise RuntimeError("Invalid cached self-attention beams")
                pieces.append(value)
            cache[key] = torch.cat(pieces, dim=0)
        m.kv_cache = cache
        if self.beam > 1:
            m.inference.kv_cache = cache
        m.dec_attns = []
        logits = m.model.decoder(batch_tokens, batch_features, kv_cache=cache)
        self.batch_sizes.append(len(group))
        output = []
        for i, item in enumerate(group):
            low, high = self.beam * i, self.beam * (i + 1)
            state = item["state"].copy()
            state["kv_cache"] = {
                key: (value[low:low + 1] if "_cross_attn_" in key else value[low:high])
                for key, value in m.kv_cache.items()
            }
            state["dec_attns"] = state["dec_attns"] + [
                x[low:high].squeeze(0) if self.beam == 1 else x[low:high]
                for x in m.dec_attns]
            output.append((item, logits[low:high], state))
        return output

    def push_many(self, chunks):
        engine = self.engine
        m = self.model
        if not chunks:
            return {}
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
                    raise RuntimeError("Unexpected decoder coroutine response")

            try:
                for cid, pcm in chunks.items():
                    session = engine.sessions[cid]
                    engine._restore(session["model"])
                    pcm = np.asarray(pcm, dtype=np.float32)
                    if pcm.ndim != 1 or not np.isfinite(pcm).all():
                        raise ValueError("Expected finite mono 16 kHz PCM")
                    session["vac"].insert_audio_chunk(pcm)
                    session["model"] = engine._snapshot()

                for cid in chunks:
                    session = engine.sessions[cid]
                    def run(call_id=cid, saved_session=session):
                        engine._restore(saved_session["model"])
                        result = saved_session["vac"].process_iter()
                        saved_session["model"] = engine._snapshot()
                        return ("done", result)
                    advance(cid, greenlet(run), first=True)

                while pending:
                    groups = defaultdict(list)
                    for item in pending.values():
                        groups[self._signature(item)].append(item)
                    pending = {}
                    decoded = []
                    for group in groups.values():
                        for i in range(0, len(group), self.max_batch):
                            decoded.extend(self._decode_group(group[i:i + self.max_batch]))
                    for item, logits, state in decoded:
                        self._restore(state)
                        advance(item["cid"], item["coroutine"], logits)
                return results
            except Exception:
                for cid in chunks:
                    engine.sessions.pop(cid, None)
                raise
            finally:
                self.active_call_id = None

    def close(self):
        self.model.logits = self.original_logits

    def finish(self, cid):
        with self.engine.lock:
            self.model.logits = self.original_logits
            try:
                return self.engine.finish(cid)
            finally:
                self.model.logits = self._pause_logits
