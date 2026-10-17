import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from .tokamax_rpa_hd64 import ragged_paged_attention_hd64, get_kv_cache_shape


def ragged_paged_attention_adapter(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    kv_pages: jnp.ndarray,
    pages_tables: jnp.ndarray,
    context_lens: jnp.ndarray,
    cu_q_lens: Optional[jnp.ndarray] = None,
    distribution: Optional[jnp.ndarray] = None,
    softmax_scale: Optional[float] = None,
    num_kv_pages_per_block: Optional[int] = 1,
    num_queries_per_block: Optional[int] = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Adapter for Tokamax Ragged Paged Attention (head_dim=64) Pallas Kernel on TPU.
    """
    batch, q_len, num_heads, dim = query.shape
    num_kv_heads = key.shape[2]

    query_ragged = query.reshape(-1, num_heads, dim)
    key_ragged = key.reshape(-1, num_kv_heads, dim)
    value_ragged = value.reshape(-1, num_kv_heads, dim)

    if cu_q_lens is None:
        cu_q_lens = jnp.arange(0, (batch + 1) * q_len, q_len, dtype=jnp.int32)
    if distribution is None:
        if q_len == 1:
            distribution = jnp.array([batch, batch, batch], dtype=jnp.int32)
        else:
            distribution = jnp.array([0, batch, batch], dtype=jnp.int32)

    if softmax_scale is None:
        softmax_scale = 1.0

    output_ragged, updated_kv_pages = ragged_paged_attention_hd64(
        queries=query_ragged,
        keys=key_ragged,
        values=value_ragged,
        kv_cache=kv_pages,
        kv_lens=context_lens,
        page_indices=pages_tables.reshape(-1),
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        sm_scale=softmax_scale,
        chunk_prefill_size=q_len if q_len > 1 else None,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
    )

    output = output_ragged.reshape(batch, q_len, num_heads, dim)
    return output, updated_kv_pages
