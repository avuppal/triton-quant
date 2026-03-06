"""
Triton kernel that dequantizes INT4 weights on-the-fly during a matrix multiply.

Memory layout
-------------
``a_packed`` stores the weight matrix A in nibble-packed format:
  - shape: (M, K // 2) of dtype uint8
  - byte[i, j] = (high_nibble << 4) | low_nibble
    where low_nibble  → A[i, 2*j]
          high_nibble → A[i, 2*j + 1]
  - Value range is unsigned [0, 15]; actual INT4 is value - 8 ∈ [-8, 7].

``scales`` is a (K,) float16 tensor; one scale per *column* of A.

The kernel unpacks each byte, dequantizes to float16, then accumulates a
standard tiled dot product with B (K × N, float16) → C (M × N, float16).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _quant_matmul_kernel(
    # Pointers
    a_packed_ptr,   # (M, K//2) uint8
    scales_ptr,     # (K,)      float16
    b_ptr,          # (K, N)    float16
    c_ptr,          # (M, N)    float16
    # Dimensions
    M: int,
    N: int,
    K: int,
    # Strides (in *elements*, not bytes)
    stride_ap_m: int,   # stride of a_packed along M  (== K//2)
    stride_b_k:  int,   # stride of b along K          (== N)
    stride_b_n:  int,   # stride of b along N          (== 1)
    stride_c_m:  int,   # stride of c along M          (== N)
    stride_c_n:  int,   # stride of c along N          (== 1)
    # Tile sizes (must be constexpr for Triton)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must be even (we process nibble pairs)
) -> None:
    """INT4-dequant fused matmul kernel.

    Each program handles one (BLOCK_M × BLOCK_N) output tile.
    Along the K dimension we iterate in steps of BLOCK_K, unpacking two
    nibbles per byte, dequantizing, and accumulating.
    """
    # --- determine which output tile this program owns --------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row / column offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --- K-dimension loop -------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)       # (BLOCK_K,)
        k_mask = offs_k < K

        # Load per-channel scales for this K block  (BLOCK_K,)
        scales = tl.load(scales_ptr + offs_k, mask=k_mask, other=1.0).to(tl.float16)

        # Load packed bytes for this tile: each byte covers two K positions
        # a_packed is indexed by (m, k//2) → stride is stride_ap_m along m,
        # 1 along the packed K axis.
        half_k_start = k_start // 2
        offs_half_k  = tl.arange(0, BLOCK_K // 2)      # (BLOCK_K//2,)
        half_k_mask  = (half_k_start + offs_half_k) < (K // 2)

        a_packed_ptrs = (
            a_packed_ptr
            + offs_m[:, None] * stride_ap_m
            + (half_k_start + offs_half_k)[None, :]
        )
        a_packed = tl.load(
            a_packed_ptrs,
            mask=((offs_m[:, None] < M) & half_k_mask[None, :]),
            other=0,
        ).to(tl.uint8)                                  # (BLOCK_M, BLOCK_K//2)

        # Unpack nibbles → signed INT4 in float16
        low  = (a_packed & 0x0F).to(tl.int16) - 8      # even k positions
        high = ((a_packed >> 4) & 0x0F).to(tl.int16) - 8  # odd  k positions

        # Re-interleave columns: [low₀, high₀, low₁, high₁, …] → (BLOCK_M, BLOCK_K)
        # Triton doesn't have interleave, so we reshape + concatenate manually.
        # low  : (BLOCK_M, BLOCK_K//2)  → even columns of a_fp16
        # high : (BLOCK_M, BLOCK_K//2)  → odd  columns of a_fp16
        a_fp16 = tl.interleave(
            low.to(tl.float16)  * tl.reshape(scales[0::2], (1, BLOCK_K // 2)),
            high.to(tl.float16) * tl.reshape(scales[1::2], (1, BLOCK_K // 2)),
        )                                               # (BLOCK_M, BLOCK_K)

        # Load B tile  (BLOCK_K, BLOCK_N)
        b_ptrs = (
            b_ptr
            + offs_k[:, None] * stride_b_k
            + offs_n[None, :] * stride_b_n
        )
        b = tl.load(
            b_ptrs,
            mask=(k_mask[:, None] & (offs_n[None, :] < N)),
            other=0.0,
        ).to(tl.float16)

        acc += tl.dot(a_fp16, b)

    # --- store output -----------------------------------------------------
    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_c_m
        + offs_n[None, :] * stride_c_n
    )
    tl.store(
        c_ptrs,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def quant_matmul(
    weight_packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` where *weight* is stored as packed INT4.

    Parameters
    ----------
    weight_packed:
        uint8 tensor of shape ``(M, K // 2)`` — packed weight matrix.
    scales:
        float16 tensor of shape ``(K,)`` — per-column dequantization scales.
    x:
        float16 tensor of shape ``(B, K)`` — input activations.
        (B is the batch / sequence length, plays the role of N.)
    block_m, block_n, block_k:
        Triton tile sizes.  block_k must be even.

    Returns
    -------
    out : torch.Tensor
        float16 tensor of shape ``(M, B)`` — matmul result.
    """
    assert weight_packed.is_cuda and x.is_cuda, "tensors must be on CUDA"
    assert block_k % 2 == 0, "block_k must be even (nibble pairs)"

    M, half_K = weight_packed.shape
    K = half_K * 2
    N = x.shape[0]  # batch dimension

    # B plays the role of the N dimension in the kernel (weight row-major)
    b_fp16 = x.contiguous()                             # (N, K)  — B matrix
    c = torch.empty((M, N), dtype=torch.float16, device=x.device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _quant_matmul_kernel[grid](
        weight_packed, scales, b_fp16.T.contiguous(), c,
        M, N, K,
        weight_packed.stride(0),
        b_fp16.T.stride(0), b_fp16.T.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return c
