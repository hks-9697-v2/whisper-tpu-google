# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokamax TPU-Friendly Ragged Paged Attention Pallas kernel optimized for head_dim = 64."""

import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
DEFAULT_VMEM_LIMIT_BYTES = 67043328


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def next_power_of_2(x: int):
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_tpu_version() -> int:
    kind = jax.devices()[0].device_kind
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
    assert kind[:-1] == "TPU v", kind
    return int(kind[-1])


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
    sliding_window=None,
):
    tpu_version = get_tpu_version()
    if tpu_version >= 7:
        bkv_p, bq = (max(1, 4096 // page_size), 32)
    else:
        bkv_p, bq = (max(1, 2048 // page_size), 32)
    return (min(pages_per_seq, bkv_p), min(max_num_tokens, bq))


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    assert actual_head_dim == 64
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads, kv_packing) // kv_packing,
        kv_packing,
        128,
    )


def merge_kv(
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
    kv = jnp.pad(
        jnp.concat([k, v], axis=-1),
        (
            (0, 0),
            (0, num_kv_heads - actual_num_kv_heads),
            (0, 0),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads // kv_packing,
        kv_packing,
        actual_head_dim * 2,
    )
    return kv


def prepare_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    attention_sink: jax.Array | None = None,
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = (
        jnp.pad(
            q.reshape(
                max_num_tokens,
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head // q_packing,
            q_packing,
            head_dim,
        )
        .swapaxes(0, 1)
    )
    kv = merge_kv(k, v)

    if attention_sink is not None:
        attention_sink = attention_sink.reshape((-1, num_q_heads_per_kv_head, 1))
        attention_sink = jnp.repeat(attention_sink, 128, -1)

    return q, kv, attention_sink


def prepare_outputs(
    out,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
):
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        actual_head_dim_x2,
    ) = out.shape
    actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
    return (
        out.swapaxes(0, 1)
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_q_packing * q_packing,
            actual_head_dim_x2,
        )[:, :, :actual_num_q_heads_per_kv_head, actual_head_dim:]
        .reshape(max_num_tokens, actual_num_q_heads, actual_head_dim)
    )


def _ragged_paged_attention_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    distribution_ref,
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    q_hbm_ref,
    kv_hbm_ref,
    kv_cache_hbm_ref,
    attention_sink_ref,
    o_hbm_ref,
    updated_kv_cache_hbm_ref,
    bkv_x2_ref,
    bq_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    bkv_p,
    bq_sz,
    debug_mode: bool = False,
):
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        actual_head_dim_x2,
    ) = q_hbm_ref.shape
    (
        total_num_pages,
        page_size,
        num_kv_heads_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_hbm_ref.shape
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    num_kv_heads = num_kv_heads_per_kv_packing * kv_packing
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    bkv_sz = bkv_p * page_size
    seq_idx = pl.program_id(0)
    num_seqs = pl.num_programs(0)
    decode_end = distribution_ref[0]
    prefill_end = distribution_ref[1]
    mixed_end = distribution_ref[2]

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]

    if sliding_window is None:
        bkv_idx_start = next_seq_bkv_idx_start = 0
    else:
        bkv_idx_start = jnp.maximum(kv_len - q_len - sliding_window, 0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, num_seqs - 1)
        next_kv_len = kv_lens_ref[next_seq_idx]
        next_q_len = cu_q_lens_ref[next_seq_idx + 1] - q_end
        next_seq_bkv_idx_start = (
            jnp.maximum(next_kv_len - next_q_len - sliding_window, 0) // bkv_sz
        )

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])], axis=-1
        )[..., : shape[-1]]

    def flash_attention_step1_qk_softmax(
        q,
        kv,
        *,
        bq_idx,
        bkv_idx,
        kv_head_idx,
    ):
        head_l_ref = l_ref.at[kv_head_idx, : q.shape[0]]
        head_m_ref = m_ref.at[kv_head_idx, : q.shape[0]]

        def load_with_init(ref, init_val):
            return jnp.where(
                bkv_idx == bkv_idx_start, jnp.full_like(ref, init_val), ref[...]
            )

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(kv.dtype, jnp.floating):
                dtype_info = jnp.finfo(kv.dtype)
                q = jnp.clip(q, min=float(dtype_info.min), max=float(dtype_info.max))
            q = q.astype(kv.dtype)

        s = jnp.matmul(q, kv.T, preferred_element_type=jnp.float32)
        s *= sm_scale
        if k_scale is not None:
            s *= k_scale
        if q_scale is not None:
            s *= q_scale
        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)

        q_span = (
            kv_len
            - q_len
            + bq_idx * bq_sz
            + lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
        )
        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, s.shape, 1)
        mask = k_span <= q_span

        if sliding_window is not None:
            mask = jnp.logical_and(mask, q_span - sliding_window < k_span)

        s = jnp.where(mask, s, mask_value)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)

        if attention_sink_ref is not None:
            sinks = attention_sink_ref[kv_head_idx]
            actual_bq_sz = q.shape[0] // num_q_heads_per_kv_head
            m_prev_init = jnp.concat([sinks] * actual_bq_sz, axis=0)
            m_prev = jnp.where(bkv_idx == bkv_idx_start, m_prev_init, head_m_ref[...])
        else:
            m_prev = load_with_init(head_m_ref, -jnp.inf)

        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = load_with_init(head_l_ref, 1.0)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr

        return p, exp_m_diff

    def flash_attention_step2_pv(
        kv,
        p,
        exp_m_diff,
        *,
        bkv_idx,
        kv_head_idx,
    ):
        head_acc_ref = acc_ref.at[kv_head_idx, : p.shape[0]]

        def load_with_init(ref, init_val):
            return jnp.where(
                bkv_idx == bkv_idx_start, jnp.full_like(ref, init_val), ref[...]
            )

        pv = jnp.matmul(p, kv, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale

        o_prev = load_with_init(head_acc_ref, 0.0)
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        if not wait:
            wait_update_kv_cache(bkv_sem_idx)
            for i in range(bkv_p):
                sz = jnp.clip(kv_left_frm_cache - i * page_size, 0, page_size)
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                _async_copy(
                    cache_hbm_ref.at[pl.ds(page_indices_ref[page_idx] * page_size, sz)],
                    vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait=False,
                )
            new_kv_len_start = q_end - kv_left_frm_new
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait,
            )
        else:
            dst = vmem_ref.at[pl.ds(0, bkv_sz_frm_cache + bkv_sz_frm_new)]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False):
        sem = sems.at[3, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )

        if not wait:
            def loop_body(i, states):
                update_sz, ignore = states
                sz = jnp.minimum(page_size - ignore, update_sz)
                _async_copy(
                    vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                    cache_hbm_ref.at[
                        pl.ds(
                            page_indices_ref[page_indices_offset + i] * page_size
                            + ignore,
                            sz,
                        )
                    ],
                    sem,
                    wait=False,
                )
                return update_sz - sz, 0

            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),
                unroll=False,
            )
        else:
            dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)
        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)
        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

    def load_bq(bq_sem_idx, kv_head_idx, *, actual_bq_sz=bq_sz):
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx, kv_head_idx]
            .reshape(
                bq_sz * num_q_heads_per_kv_head_per_packing,
                actual_head_dim_x2,
            )
        )
        return pltpu.bitcast(
            q_ref[: actual_bq_sz * num_q_heads_per_kv_head_per_packing], q_dtype
        )

    def strided_load(ref, start, step):
        return ref[start::step]

    def strided_load_bkv(bkv_sem_idx, start, step):
        start //= kv_packing
        step //= kv_packing
        kv_ref = (
            bkv_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx]
            .reshape(bkv_sz * step, actual_head_dim_x2)
        )
        kv = strided_load(kv_ref, start, step)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        lst = []
        for i in range(0, kv_packing):
            cur_kv = pltpu.bitcast((kv >> (i * bitwidth)).astype(repack_ty), kv_dtype)
            lst.append(cur_kv)
        return lst

    def process(static_q_len=None):
        num_bkv = cdiv(kv_len, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)

            if sliding_window is None:
                next_bkv_idx_start = 0
            else:
                next_bkv_idx_start = lax.select(
                    is_last_bq, next_seq_bkv_idx_start, bkv_idx_start
                )
            next_bkv_idx = lax.select(is_last_bkv, next_bkv_idx_start, next_bkv_idx)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            @pl.when(next_seq_idx < num_seqs)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx
                )

                @pl.when(next_seq_idx < num_seqs)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

                @pl.when(bkv_idx == bkv_idx_start)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == 0))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                if debug_mode:
                    return

                prev_kv_head_bkv = None
                prev_kv_head_idx = None
                prev_kv_head_p = None
                prev_kv_head_exp_m_diff = None
                for kv_head_start in range(0, actual_num_kv_heads, kv_packing):
                    bkv_lst = strided_load_bkv(
                        bkv_sem_idx,
                        kv_head_start,
                        num_kv_heads,
                    )
                    for i in range(kv_packing):
                        cur_kv_head_idx = kv_head_start + i
                        if cur_kv_head_idx >= actual_num_kv_heads:
                            break
                        cur_kv_head_bq = load_bq(
                            bq_sem_idx, cur_kv_head_idx, actual_bq_sz=actual_bq_sz
                        )
                        cur_kv_head_bkv = bkv_lst[i]
                        cur_kv_head_p, cur_kv_head_exp_m_diff = (
                            flash_attention_step1_qk_softmax(
                                cur_kv_head_bq,
                                cur_kv_head_bkv,
                                bq_idx=bq_idx,
                                bkv_idx=bkv_idx,
                                kv_head_idx=cur_kv_head_idx,
                            )
                        )
                        if prev_kv_head_p is not None:
                            flash_attention_step2_pv(
                                prev_kv_head_bkv,
                                prev_kv_head_p,
                                prev_kv_head_exp_m_diff,
                                bkv_idx=bkv_idx,
                                kv_head_idx=prev_kv_head_idx,
                            )
                        prev_kv_head_bkv = cur_kv_head_bkv
                        prev_kv_head_p = cur_kv_head_p
                        prev_kv_head_exp_m_diff = cur_kv_head_exp_m_diff
                        prev_kv_head_idx = cur_kv_head_idx

                flash_attention_step2_pv(
                    prev_kv_head_bkv,
                    prev_kv_head_p,
                    prev_kv_head_exp_m_diff,
                    bkv_idx=bkv_idx,
                    kv_head_idx=prev_kv_head_idx,
                )

            lax.fori_loop(bkv_idx_start, num_bkv, compute_with_bkv, None, unroll=False)

            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)
            out = lax.div(acc, l).astype(q_dtype)

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                actual_num_kv_heads,
                bq_sz * num_q_heads_per_kv_head_per_packing,
                actual_head_dim_x2,
            )[...] = pltpu.bitcast(out, jnp.int32)

            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    @pl.when(seq_idx == 0)
    def prologue():
        start_fetch_bq(0, 0, 0)
        bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32).reshape((2, -1, 8, 128))
        zeros = jnp.zeros(bkv_x2_int32_ref.shape[1:], jnp.int32)
        bkv_x2_int32_ref[0] = zeros
        start_fetch_bkv(0, bkv_idx_start, 0)
        bkv_x2_int32_ref[1] = zeros

    @pl.when(seq_idx < decode_end)
    def process_decode():
        process(static_q_len=1)

    @pl.when(jnp.logical_and(decode_end <= seq_idx, seq_idx < prefill_end))
    def process_prefill():
        process(static_q_len=chunk_prefill_size)

    @pl.when(jnp.logical_and(prefill_end <= seq_idx, seq_idx < mixed_end))
    def process_mixed():
        process()

    @pl.when(seq_idx == num_seqs - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)


def ragged_paged_attention_hd64(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, 64]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, 64]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, 64]
    kv_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads // kv_packing, kv_packing, 128]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sink: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
):
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    q, k, v = queries, keys, values
    actual_num_q_heads = q.shape[1]
    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q, kv, attention_sink = prepare_inputs(q, k, v, attention_sink)
    (
        _,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing

    bkv_p = num_kv_pages_per_block
    bq_sz = num_queries_per_block
    if bq_sz is None or bkv_p is None:
        bkv_p, bq_sz = get_tuned_block_sizes(
            q.dtype,
            kv_cache.dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            actual_head_dim,
            page_size,
            max_num_tokens,
            pages_per_seq,
            sliding_window,
        )
    bkv_sz = bkv_p * page_size
    if vmem_limit_bytes is None:
        vmem_limit_bytes = DEFAULT_VMEM_LIMIT_BYTES
    grid = (max_num_seqs,)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        None if attention_sink is None else pl.BlockSpec(memory_space=pltpu.VMEM),
    ]
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]

    bkv_double_buf = pltpu.VMEM((2, bkv_sz, *kv_cache.shape[2:]), kv_cache.dtype)
    bq_double_buf = pltpu.VMEM((2, actual_num_kv_heads, bq_sz, *q.shape[2:]), q.dtype)
    bo_double_buf = bq_double_buf

    l_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
        jnp.float32,
    )
    m_scratch = l_scratch
    acc_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
        jnp.float32,
    )

    scratch_shapes = [
        bkv_double_buf,
        bq_double_buf,
        bo_double_buf,
        pltpu.SemaphoreType.DMA((4, 2)),
        l_scratch,
        m_scratch,
        acc_scratch,
    ]

    scalar_prefetches = (
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        jnp.zeros((3,), jnp.int32),
        jnp.full((4,), -1, jnp.int32),
        jnp.full((6,), -1, jnp.int32),
    )

    scope_name = f"RPA-HD_64-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
    kernel = pl.pallas_call(
        functools.partial(
            _ragged_paged_attention_kernel,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            chunk_prefill_size=chunk_prefill_size,
            bq_sz=bq_sz,
            bkv_p=bkv_p,
            debug_mode=debug_mode,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        out_shape=[
            jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=kv_cache.shape, dtype=kv_cache.dtype),
        ],
        input_output_aliases={
            7: 0,
            9: 1,
        },
        name=scope_name,
    )

    output, updated_kv_cache = kernel(
        *scalar_prefetches, q, kv, kv_cache, attention_sink
    )
    return (
        prepare_outputs(output, actual_num_q_heads_per_kv_head, actual_head_dim),
        updated_kv_cache,
    )
