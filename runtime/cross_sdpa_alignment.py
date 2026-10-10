"""Experimental fused decoder cross-attention retaining AlignAtt head logits."""
import torch
import torch.nn.functional as F


def install(model):
    blocks = model.model.decoder.blocks
    original_sources = {rank: list(entries) for rank, entries in model.align_source.items()}
    for rank, block in enumerate(blocks):
        module = block.cross_attn
        hooks = module._forward_hooks
        if len(hooks) != 1:
            raise ValueError(f"Expected one cross-attention hook in layer {rank}")
        hooks.clear()
        sources = original_sources.get(rank, [])
        selected = list(dict.fromkeys(int(head) for _, head in sources))
        compact = {head: i for i, head in enumerate(selected)}
        model.align_source[rank] = [(source_rank, compact[int(head)])
                                    for source_rank, head in sources]
        selected_gpu = torch.tensor(selected, dtype=torch.long, device=model.model.device)

        def attention(q, k, v, mask=None, indices=selected_gpu, n_head=module.n_head):
            if mask is not None:
                raise ValueError("Cross-attention unexpectedly received a mask")
            batch, n_ctx, n_state = q.shape
            q = q.view(batch, n_ctx, n_head, -1).permute(0, 2, 1, 3)
            k = k.view(k.shape[0], k.shape[1], n_head, -1).permute(0, 2, 1, 3)
            v = v.view(v.shape[0], v.shape[1], n_head, -1).permute(0, 2, 1, 3)
            output = F.scaled_dot_product_attention(q, k, v)
            output = output.permute(0, 2, 1, 3).flatten(start_dim=2)
            if indices.numel():
                scale = (n_state // n_head) ** -0.25
                selected_q = q.index_select(1, indices)
                selected_k = k.index_select(1, indices)
                qk = ((selected_q * scale) @
                      (selected_k * scale).transpose(-1, -2)).float().detach()
            else:
                qk = torch.empty((batch, 0, n_ctx, k.shape[2]),
                                 dtype=torch.float32, device=q.device)
            return output, qk

        def hook(_module, _inputs, output):
            model.dec_attns.append(F.softmax(output[1], dim=-1).squeeze(0))

        module.qkv_attention = attention
        module.register_forward_hook(hook)
    return {"layers": len(blocks), "selected_heads": sum(map(len, original_sources.values()))}
