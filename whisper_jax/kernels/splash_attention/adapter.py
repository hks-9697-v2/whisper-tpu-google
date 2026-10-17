import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from typing import Optional

from .splash_attention_kernel import make_splash_mha, SplashConfig, BlockSizes, QKVLayout
from .splash_attention_mask import FullMask, CausalMask, NumpyMask, LogicalAnd

_SPLASH_STATIC_CACHE = {}


@partial(jax.jit, static_argnames=("q_seq_len", "kv_seq_len", "mask", "config", "scale_query"))
def _splash_attention_kernel(query, key, value, mask, q_seq_len, kv_seq_len, config, scale_query: bool = True):
    """
    A JIT-compiled kernel function for the latest Tokamax Splash Attention.
    """
    head_dim = query.shape[-1]

    # --- 1. Scale Query ---
    if scale_query:
        query = query * (head_dim ** -0.5)

    # --- 2. Handle Sequence Padding on axis 1 before transpose so XLA fuses transposes ---
    BLOCK_SIZE = 128
    SEQ_AXIS = 1

    def pad_to_multiple(x, seq_len):
        pad_len = (BLOCK_SIZE - seq_len % BLOCK_SIZE) % BLOCK_SIZE
        if pad_len == 0:
            return x
        pad_width_seq = [(0, 0)] * x.ndim
        pad_width_seq[SEQ_AXIS] = (0, pad_len)
        return jnp.pad(x, pad_width_seq)

    q_padded = pad_to_multiple(query, q_seq_len)
    k_padded = pad_to_multiple(key, kv_seq_len)
    v_padded = pad_to_multiple(value, kv_seq_len)

    # --- 3. Transpose for Kernel: (B, S, H, D) -> (B, H, S, D) ---
    q_padded_seq = q_padded.transpose(0, 2, 1, 3)
    k_padded_seq = k_padded.transpose(0, 2, 1, 3)
    v_padded_seq = v_padded.transpose(0, 2, 1, 3)

    # --- 4. Create and Execute the Kernel ---
    splash_mha_kernel = make_splash_mha(
        mask=mask,
        config=config,
        q_seq_shards=1,
        downcast_smem_data=False,
    )
    vmapped_kernel = jax.vmap(splash_mha_kernel, in_axes=(0, 0, 0), out_axes=0)

    output_padded = vmapped_kernel(q_padded_seq, k_padded_seq, v_padded_seq)

    # --- 5. Slice Padding and Reshape Output Back ---
    output_sliced_seq = output_padded[:, :, :q_seq_len, :head_dim]
    output = output_sliced_seq.transpose(0, 2, 1, 3)

    return output


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


def _get_tuned_splash_block_sizes(q_seq_len: int) -> tuple[int, int, int]:
    """Returns hardware-tuned (block_q, block_kv, block_kv_compute) for QKVLayout.SEQ_MINOR."""
    tpu_version = _get_tpu_version()
    if tpu_version >= 7:
        # TPU v7x (64MB VMEM): full-sequence Q/KV block with 768 compute sub-tile (0.795ms)
        return (1536, 1536, 768)
    elif tpu_version == 6:
        # TPU v6e (32MB VMEM): split Q into 768 blocks with 512 compute sub-tile (1.738ms)
        return (768, 1536, 512)
    else:
        return (768, 1536, 512)


def _get_default_splash_config(q_seq_len: int) -> SplashConfig:
    if q_seq_len <= 128:
        return SplashConfig(
            block_q=128,
            block_kv=128,
            block_kv_compute=128,
            block_q_dkv=64,
            block_kv_dkv=64,
            block_kv_dkv_compute=64,
            q_layout=QKVLayout.SEQ_MINOR,
            k_layout=QKVLayout.SEQ_MINOR,
            v_layout=QKVLayout.SEQ_MINOR,
            use_fused_bwd_kernel=True,
            use_base2_exp=True,
            fuse_reciprocal=True,
            use_experimental_scheduler=False,
        )
    else:
        block_q, block_kv, block_kv_compute = _get_tuned_splash_block_sizes(q_seq_len)
        return SplashConfig(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            block_q_dkv=128,
            block_kv_dkv=1536,
            block_kv_dkv_compute=384,
            q_layout=QKVLayout.SEQ_MINOR,
            k_layout=QKVLayout.SEQ_MINOR,
            v_layout=QKVLayout.SEQ_MINOR,
            use_fused_bwd_kernel=True,
            use_base2_exp=True,
            fuse_reciprocal=True,
            use_experimental_scheduler=False,
        )


def splash_attention_adapter(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray] = None,
    causal: bool = False,
    scale_query: bool = True,
    config: Optional[SplashConfig] = None,
):
    """
    A general splash attention adapter using the latest Tokamax Splash Attention kernel.
    """
    batch_size, q_seq_len, num_heads, head_dim = query.shape
    _, kv_seq_len, _, _ = key.shape

    BLOCK_SIZE = 128
    padded_q_len = (q_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    padded_kv_len = (kv_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    mask_shape = (padded_q_len, padded_kv_len)

    if attention_mask is None:
        cache_key = (q_seq_len, kv_seq_len, causal, config)
        cached = _SPLASH_STATIC_CACHE.get(cache_key)
        if cached is not None:
            mask, splash_config = cached
        else:
            mask = CausalMask(shape=mask_shape) if causal else FullMask(mask_shape)
            splash_config = config if config is not None else _get_default_splash_config(q_seq_len)
            _SPLASH_STATIC_CACHE[cache_key] = (mask, splash_config)
        return _splash_attention_kernel(
            query, key, value, mask, q_seq_len, kv_seq_len, splash_config, scale_query=scale_query
        )

    attention_mask = np.asarray(attention_mask)
    if attention_mask.ndim == 2:
        attention_mask = np.expand_dims(attention_mask, axis=1)
    attention_mask = np.logical_not(attention_mask)
    padded_attention_mask = np.pad(
        attention_mask,
        ((0, padded_q_len - q_seq_len), (0, padded_kv_len - kv_seq_len)),
        mode="constant",
        constant_values=False,
    )
    padding_mask = NumpyMask(padded_attention_mask)
    if causal:
        causal_mask = CausalMask(shape=mask_shape)
        mask = LogicalAnd(causal_mask, padding_mask)
    else:
        mask = padding_mask

    splash_config = config if config is not None else _get_default_splash_config(q_seq_len)
    return _splash_attention_kernel(
        query, key, value, mask, q_seq_len, kv_seq_len, splash_config, scale_query=scale_query
    )
