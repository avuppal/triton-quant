"""
triton_int4 — INT4 weight quantization utilities and Triton dequant kernels.

Public API
----------
quantize_int4      : quantize a float16 tensor to packed INT4 + per-channel scales
dequantize_int4    : reconstruct float16 from packed INT4 + scales (CPU / GPU)
quant_matmul       : GPU matmul that dequantizes on-the-fly inside the kernel
"""

from .quant import quantize_int4, dequantize_int4  # noqa: F401

try:
    from .kernel import quant_matmul  # noqa: F401  (requires CUDA + Triton)
except Exception:
    pass  # Triton unavailable; quant_matmul not exposed
