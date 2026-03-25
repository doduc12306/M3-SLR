import torch
import torch.nn.functional as F


def _causal_mask(seq_len_q, seq_len_k, device):
    return torch.triu(
        torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=device),
        diagonal=1,
    )


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """
    CPU/GPU fallback implementation for projects that import FlashAttention API.
    Input shape: [B, N, 3, H, D]
    Output shape: [B, N, H, D]
    """
    if qkv.dim() != 5 or qkv.size(2) != 3:
        raise ValueError(f"Expected qkv shape [B, N, 3, H, D], got {tuple(qkv.shape)}")

    q, k, v = qkv.unbind(dim=2)  # [B, N, H, D]

    # Compute in float32 on CPU for stability/compatibility.
    compute_dtype = q.dtype
    if q.device.type == 'cpu' and q.dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    q = q.to(compute_dtype).permute(0, 2, 1, 3)  # [B, H, N, D]
    k = k.to(compute_dtype).permute(0, 2, 1, 3)  # [B, H, N, D]
    v = v.to(compute_dtype).permute(0, 2, 1, 3)  # [B, H, N, D]

    scale = softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

    if causal:
        mask = _causal_mask(attn_scores.shape[-2], attn_scores.shape[-1], attn_scores.device)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

    attn = torch.softmax(attn_scores, dim=-1)
    if dropout_p and dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=True)

    out = torch.matmul(attn, v)  # [B, H, N, D]
    out = out.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]
    return out.to(qkv.dtype)


def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("flash_attn_func fallback is not implemented; use flash_attn_qkvpacked_func.")