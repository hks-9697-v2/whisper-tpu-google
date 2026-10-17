import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .flash_attn_interface import flash_attn_with_kvcache
from .pallas.block_sizes import LOG2E, BlockSizes, TokenMajorInfo
from .pallas.flash_fwd import make_flash_attn_mha


def _get_tpu_version() -> int:
    try:
        kind = jax.devices()[0].device_kind
    except Exception:
        return -1
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    if kind.endswith("p") or kind.endswith("e") or kind.endswith("i"):
        kind = kind[:-1]
    if kind.startswith("TPU7"):
        return 7
    if kind.startswith("TPU8"):
        return 8
    if kind.startswith("TPU v") and kind[-1].isdigit():
        return int(kind[-1])
    return -1


def _get_tuned_flywheel_block_sizes(padded_seq_len: int) -> Tuple[int, int, int, int]:
    """Returns hardware-tuned (block_q, block_kv, block_q_compute, block_kv_compute) for FlyWheel forward MHA."""
    if padded_seq_len <= 128:
        return (128, 128, 64, 128)
    tpu_version = _get_tpu_version()
    if tpu_version >= 7:
        # TPU v7x (64MB VMEM): full-sequence Q/KV block with (256, 512) compute tiles (1.052ms at B=10)
        return (padded_seq_len, padded_seq_len, 256, 512)
    elif tpu_version == 6:
        # TPU v6e: full-sequence Q/KV block with (512, 512) compute tiles (2.068ms at B=20)
        return (padded_seq_len, padded_seq_len, 512, 512)
    else:
        return (padded_seq_len, padded_seq_len, 256, 512)


@functools.cache
def _get_flywheel_encoder_kernel(
    batch_size: int,
    padded_q_len: int,
    padded_kv_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    block_sizes_tuple: Tuple[int, int, int, int],
    q_scale: float,
):
    bq, bkv, bqc, bkvc = block_sizes_tuple
    bs = BlockSizes(
        block_q=bq,
        block_kv=bkv,
        block_q_compute=bqc,
        block_kv_compute=bkvc,
        num_stages=2,
    )
    tm_info = TokenMajorInfo(
        batch=batch_size,
        num_q_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim_qk=head_dim,
        head_dim_v=head_dim,
    )
    head_fold = 2 if (head_dim == 64 and num_heads % 2 == 0) else 1
    return make_flash_attn_mha(
        num_heads,
        padded_q_len,
        padded_kv_len,
        causal=causal,
        block_sizes=bs,
        num_kv_heads=num_heads,
        head_fold=head_fold,
        token_major=tm_info,
        q_scale=q_scale,
    )


def flywheel_encoder_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    causal: bool = False,
    scale_query: bool = True,
    block_sizes: Optional[Tuple[int, int, int, int]] = None,
) -> jnp.ndarray:
    """
    FlyWheel TPU Pallas Flash Attention adapter for Whisper encoder.
    Operates directly in token-major (B, T, H * D) layout with paired d=64 head DMAs.
    """
    batch_size, q_seq_len, num_heads, head_dim = query.shape
    _, kv_seq_len, _, _ = key.shape
    total_dim = num_heads * head_dim

    block_align = 128
    padded_q_len = ((q_seq_len + block_align - 1) // block_align) * block_align
    padded_kv_len = ((kv_seq_len + block_align - 1) // block_align) * block_align

    if block_sizes is None:
        block_sizes = _get_tuned_flywheel_block_sizes(padded_q_len)

    sm_scale = (head_dim ** -0.5) if scale_query else 1.0
    q_scale = float(sm_scale * LOG2E)

    q_tm = query.reshape(batch_size, q_seq_len, total_dim)
    k_tm = key.reshape(batch_size, kv_seq_len, total_dim)
    v_tm = value.reshape(batch_size, kv_seq_len, total_dim)

    if padded_q_len > q_seq_len:
        q_tm = jnp.pad(q_tm, ((0, 0), (0, padded_q_len - q_seq_len), (0, 0)))
    if padded_kv_len > kv_seq_len:
        pad_kv = ((0, 0), (0, padded_kv_len - kv_seq_len), (0, 0))
        k_tm = jnp.pad(k_tm, pad_kv)
        v_tm = jnp.pad(v_tm, pad_kv)

    kernel = _get_flywheel_encoder_kernel(
        batch_size=batch_size,
        padded_q_len=padded_q_len,
        padded_kv_len=padded_kv_len,
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        block_sizes_tuple=block_sizes,
        q_scale=q_scale,
    )
    out_padded = kernel(q_tm, k_tm, v_tm)
    return out_padded[:, :q_seq_len, :].reshape(batch_size, q_seq_len, num_heads, head_dim)


def flywheel_kvcache_attention(
    query: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: Optional[jnp.ndarray] = None,
    key: Optional[jnp.ndarray] = None,
    value: Optional[jnp.ndarray] = None,
    cache_seqlens: jnp.ndarray | int = 0,
    block_table: Optional[jnp.ndarray] = None,
    softmax_scale: Optional[float] = 1.0,
    causal: bool = True,
):
    """
    FlyWheel TPU Pallas KV-cache decode attention adapter.
    Supports both contiguous token-major (B, capacity, H, D) caches (v_cache provided, block_table=None)
    and paged merged (num_pages, page_size, 2*H, D) caches (v_cache=None, block_table provided).
    """
    return flash_attn_with_kvcache(
        q=query,
        k_cache=k_cache,
        v_cache=v_cache,
        k=key,
        v=value,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
    )
