"""
Triton AOT compilation: FlashAttention-style forward attention.

Computes O = softmax((Q @ K^T) * sm_scale) @ V without materializing the
full scores or probability matrices.  The tensor contract is 4D
[BATCH, HEADS, SEQ, HEAD_DIM], and the launch grid follows the Triton fused
attention tutorial shape: program_id(0) selects the query tile and
program_id(1) selects the flattened batch/head lane.
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


BATCH = env_int("FLASH_ATTENTION_BATCH")
HEADS = env_int("FLASH_ATTENTION_HEADS")
SEQ = env_int("FLASH_ATTENTION_SEQ")
HEAD_DIM = env_int("FLASH_ATTENTION_HEAD_DIM")
BLOCK_M = env_int("FLASH_ATTENTION_BLOCK_M")
BLOCK_N = env_int("FLASH_ATTENTION_BLOCK_N")
CAUSAL = env_int("FLASH_ATTENTION_CAUSAL")

if BATCH <= 0 or HEADS <= 0:
    raise ValueError("flash_attention requires positive BATCH and HEADS")
if BLOCK_M <= 0 or BLOCK_N <= 0:
    raise ValueError("flash_attention requires positive block sizes")
if SEQ % BLOCK_M != 0 or SEQ % BLOCK_N != 0:
    raise ValueError("flash_attention requires SEQ to be divisible by BLOCK_M and BLOCK_N")
if HEAD_DIM % 16 != 0:
    raise ValueError("flash_attention HEAD_DIM must be a multiple of 16")
if BLOCK_N > HEAD_DIM:
    raise ValueError("flash_attention keeps BLOCK_N <= HEAD_DIM for this CPU fixture")
if CAUSAL and BLOCK_M % BLOCK_N != 0:
    raise ValueError("flash_attention causal stage 2 requires BLOCK_M to be divisible by BLOCK_N")
if CAUSAL not in (0, 1):
    raise ValueError("flash_attention CAUSAL must be 0 or 1")

GRID_X = triton.cdiv(SEQ, BLOCK_M)
GRID_Y = BATCH * HEADS


@triton.jit
def _flash_attention_inner(acc, l_i, m_i, q,
                           k_block_ptr, v_block_ptr,
                           start_m, offs_m: tl.constexpr,
                           offs_n: tl.constexpr, sm_scale,
                           STAGE: tl.constexpr,
                           SEQ: tl.constexpr, HEAD_DIM: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    neg_inf = -3.4028234663852886e38
    if STAGE == 1:
        lo = 0
        hi = start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo = 0
        hi = SEQ

    k_block_ptr = tl.advance(k_block_ptr, (0, lo))
    v_block_ptr = tl.advance(v_block_ptr, (lo, 0))
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n = start_n + offs_n
        k = tl.load(k_block_ptr).to(tl.float32)
        v = tl.load(v_block_ptr).to(tl.float32)

        qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale
        if STAGE == 2:
            qk = tl.where(offs_m[:, None] >= n[None, :], qk, neg_inf)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_ij = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float32)
        m_i = m_ij
        l_i = l_ij
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def flash_attention(q_ptr, k_ptr, v_ptr, out_ptr, sm_scale,
                    BATCH: tl.constexpr, HEADS: tl.constexpr,
                    SEQ: tl.constexpr, HEAD_DIM: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // HEADS
    off_h = off_hz - off_b * HEADS

    base = ((off_b * HEADS + off_h) * SEQ) * HEAD_DIM
    row_base = off_hz * SEQ
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(BATCH * HEADS * SEQ, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(row_base + pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr).to(tl.float32)

    neg_inf = -3.4028234663852886e38
    m_i = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(HEAD_DIM, BATCH * HEADS * SEQ),
        strides=(1, HEAD_DIM),
        offsets=(0, row_base),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(BATCH * HEADS * SEQ, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(row_base, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    if CAUSAL:
        acc, l_i, m_i = _flash_attention_inner(
            acc, l_i, m_i, q, k_block_ptr, v_block_ptr,
            pid_m, offs_m, offs_n, sm_scale, 1,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N,
        )
        acc, l_i, m_i = _flash_attention_inner(
            acc, l_i, m_i, q, k_block_ptr, v_block_ptr,
            pid_m, offs_m, offs_n, sm_scale, 2,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N,
        )
    else:
        acc, l_i, m_i = _flash_attention_inner(
            acc, l_i, m_i, q, k_block_ptr, v_block_ptr,
            pid_m, offs_m, offs_n, sm_scale, 3,
            SEQ, HEAD_DIM, BLOCK_M, BLOCK_N,
        )

    out = acc / l_i[:, None]
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(BATCH * HEADS * SEQ, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(row_base + pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(out_block_ptr, out)


q = torch.empty(BATCH, HEADS, SEQ, HEAD_DIM, dtype=torch.float32)
k = torch.empty(BATCH, HEADS, SEQ, HEAD_DIM, dtype=torch.float32)
v = torch.empty(BATCH, HEADS, SEQ, HEAD_DIM, dtype=torch.float32)
out = torch.empty(BATCH, HEADS, SEQ, HEAD_DIM, dtype=torch.float32)

flash_attention[(GRID_X, GRID_Y)](
    q, k, v, out, 1.0, BATCH, HEADS, SEQ, HEAD_DIM, BLOCK_M, BLOCK_N, CAUSAL
)
