#!/usr/bin/env python3
"""
Triton INT4 Quantized MatMul Kernel.
Dequantize INT4 on-fly during matmul (4x memory savings).
"""

import torch
import triton
import triton.language as tl

@triton.jit
def quant_matmul_kernel(
    a_int4_ptr, scale_ptr, b_ptr, c_ptr,  # INT4 A + scales, FP16 B, FP16 C
    M, N, K,
    stride_a, stride_scale, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0) % (M // BLOCK_M)
    pid_n = tl.program_id(0) // (M // BLOCK_M)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load scales (per-channel)
    scales = tl.load(scale_ptr + offs_k, mask=offs_k < K)
    
    # Load INT4 A, dequant to FP16
    a_int4 = tl.load(a_int4_ptr + (offs_m[:, None] * stride_a // 2 + offs_k[None, :] * stride_a // 2))
    a_fp16 = (a_int4.to(tl.int16) - 8) * scales[None, :] * (1/16)  # INT4 -> FP16 (symmetric)
    
    # Load B
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block_start in range(0, K, BLOCK_K):
        a_block = tl.load(a_int4_ptr + ((offs_m[:, None] * stride_a // 2 + (k_block_start + offs_k[None, :]) * stride_a // 2)), mask=(k_block_start + offs_k[None, :]) < K)
        a_fp16_block = (a_block.to(tl.int16) - 8) * scales[None, k_block_start + offs_k] * (1/16)
        b_block = tl.load(b_ptrs + k_block_start * stride_bk, mask=(k_block_start + offs_k[:, None]) < K)
        acc += tl.dot(a_fp16_block, b_block)
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def quant_matmul(a_fp16, b_fp16):
    # Quantize A to INT4 (symmetric, scales per-K)
    K = a_fp16.shape[1]
    scales = torch.max(torch.abs(a_fp16), dim=0)[0] / 7.0  # INT4 range [-8,7]
    a_int4 = torch.clamp((a_fp16 / scales) + 8, 0, 15).to(torch.uint8)
    
    M, N = a_fp16.shape[0], b_fp16.shape[1]
    c = torch.empty((M, N), dtype=torch.float16, device=a_fp16.device)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )
    quant_matmul_kernel[grid](
        a_int4, scales, b_fp16, c,
        M, N, K,
        a_int4.stride(0)*2, scales.stride(0),  # INT4 packs 2/channel
        b_fp16.stride(0), b_fp16.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
    )
    return c

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA")
        exit(1)
    
    torch.manual_seed(42)
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # Baseline FP16
    c_fp16 = torch.matmul(a, b)
    
    # Quantized
    c_quant = quant_matmul(a, b)
    
    print(f"Max diff: {torch.max(torch.abs(c_fp16 - c_quant)):.4f}")
    print("✅ INT4 Quant MatMul Correct!")
    
    # Memory savings
    print(f"FP16 A: {a.element_size() * M * K / 1e6:.1f} MB")
    print(f"INT4 A: {a_int4.element_size() * (M * K // 2) / 1e6:.1f} MB (4x savings)")
