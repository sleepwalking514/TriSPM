"""
Triton AOT compilation: single-query paged KV-decode kernel.

Minimal PagedAttention-style decode microkernel:

- one batch item, one head, one query vector ``q``;
- fixed ``NUM_PAGES`` / ``PAGE_SIZE`` / ``HEAD_DIM`` / ``NUM_PHYS_PAGES``;
- page table ``page_ids[NUM_PAGES]`` of int32 physical-page indices;
- K/V caches laid out as contiguous physical pages
  ``[NUM_PHYS_PAGES, PAGE_SIZE, HEAD_DIM]``;
- exact attention via online (FlashAttention-style) softmax across the
  ``NUM_PAGES * PAGE_SIZE`` tokens selected by the page table.

The kernel itself does not call any SPM intrinsic.  Default builds are still
the cache path; opt-in compiler experiments in ConvertMemoryToSPM can recognize
the page-gather idiom and stage K/V pages into SPM.

The first irregular axis is the per-page ``phys = page_ids[p]`` lookup; the
K/V tile within a page is still contiguous and large enough for the
existing 2D DMA to stage as a single coarse descriptor, which is the
distinguishing premise from the embedding_bag probe.
"""
import os

import torch
import triton
import triton.language as tl


def env_int(name: str) -> int:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(
            f"{name} must be exported from experiment.toml by run_experiment.py"
        )
    return int(value)


NUM_PAGES = env_int("PAGED_KV_DECODE_NUM_PAGES")
NUM_PHYS_PAGES = env_int("PAGED_KV_DECODE_NUM_PHYS_PAGES")
PAGE_SIZE = env_int("PAGED_KV_DECODE_PAGE_SIZE")
HEAD_DIM = env_int("PAGED_KV_DECODE_HEAD_DIM")

if NUM_PAGES <= 0 or NUM_PHYS_PAGES <= 0:
    raise ValueError("paged_kv_decode requires positive NUM_PAGES and NUM_PHYS_PAGES")
if PAGE_SIZE <= 0 or HEAD_DIM <= 0:
    raise ValueError("paged_kv_decode requires positive PAGE_SIZE and HEAD_DIM")
if NUM_PHYS_PAGES < NUM_PAGES:
    raise ValueError(
        "paged_kv_decode requires NUM_PHYS_PAGES >= NUM_PAGES so the page "
        "table can select a non-trivial physical layout"
    )


@triton.jit
def paged_kv_decode(q_ptr, k_cache_ptr, v_cache_ptr, page_ids_ptr, out_ptr,
                    sm_scale,
                    NUM_PAGES: tl.constexpr,
                    NUM_PHYS_PAGES: tl.constexpr,
                    PAGE_SIZE: tl.constexpr,
                    HEAD_DIM: tl.constexpr):
    neg_inf = -3.4028234663852886e38

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr, shape=(HEAD_DIM,), strides=(1,),
        offsets=(0,), block_shape=(HEAD_DIM,), order=(0,))
    q = tl.load(q_block_ptr).to(tl.float32)

    m_i = tl.full((1,), neg_inf, dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for p in range(0, NUM_PAGES):
        phys = tl.load(page_ids_ptr + p)

        k_page_ptr = tl.make_block_ptr(
            base=k_cache_ptr,
            shape=(NUM_PHYS_PAGES * PAGE_SIZE, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(phys * PAGE_SIZE, 0),
            block_shape=(PAGE_SIZE, HEAD_DIM),
            order=(1, 0))
        v_page_ptr = tl.make_block_ptr(
            base=v_cache_ptr,
            shape=(NUM_PHYS_PAGES * PAGE_SIZE, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(phys * PAGE_SIZE, 0),
            block_shape=(PAGE_SIZE, HEAD_DIM),
            order=(1, 0))

        k_page = tl.load(k_page_ptr).to(tl.float32)
        v_page = tl.load(v_page_ptr).to(tl.float32)

        scores = tl.sum(k_page * q[None, :], axis=1) * sm_scale

        tile_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, tile_max)
        alpha = tl.exp(m_i - m_new)
        p_block = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p_block, axis=0)
        acc = acc * alpha + tl.sum(p_block[:, None] * v_page, axis=0)
        m_i = m_new

    out = acc / l_i
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(HEAD_DIM,), strides=(1,),
        offsets=(0,), block_shape=(HEAD_DIM,), order=(0,))
    tl.store(out_block_ptr, out)


# --- AOT cross-compilation ---
q = torch.empty(HEAD_DIM, dtype=torch.float32)
k_cache = torch.empty(NUM_PHYS_PAGES * PAGE_SIZE, HEAD_DIM, dtype=torch.float32)
v_cache = torch.empty(NUM_PHYS_PAGES * PAGE_SIZE, HEAD_DIM, dtype=torch.float32)
page_ids = torch.empty(NUM_PAGES, dtype=torch.int32)
out = torch.empty(HEAD_DIM, dtype=torch.float32)

paged_kv_decode[(1,)](q, k_cache, v_cache, page_ids, out, 1.0,
                      NUM_PAGES, NUM_PHYS_PAGES, PAGE_SIZE, HEAD_DIM)
