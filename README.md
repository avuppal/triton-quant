# triton-quant

**INT4 weight quantization and dequantization-fused Triton kernels.**

Runs matrix multiplications where the weight matrix is stored at 4-bit (INT4) precision and dequantized *on the fly* during the kernel — achieving a **4× memory reduction** with minimal accuracy loss.

---

## Why INT4?

Large language models and vision transformers are memory-bandwidth-bound during inference. The weight matrix of a linear layer is the dominant consumer:

| Format  | Bits/element | Memory for 7B params |
|---------|-------------|----------------------|
| float32 | 32          | 28 GB                |
| float16 | 16          | 14 GB                |
| INT8    | 8           | 7 GB                 |
| **INT4**| **4**       | **3.5 GB**           |

Moving from float16 → INT4 cuts the weight footprint by **4×**, allowing a 14 GB model to fit in a single 16 GB GPU — or doubling the batch size at the same memory budget.

### The trade-off

INT4 has 16 representable levels. Per-channel symmetric quantization maps the continuous range `[-max_col, max_col]` to `{-8, -7, …, 7}`:

```
scale  = max|W[:, k]| / 7          (one scale per column)
q      = round(W[:, k] / scale)    ∈ [-8, 7]
packed = q + 8                     ∈ [0, 15]  (stored as a nibble)
```

Two nibbles are packed into a single `uint8` byte, giving the 4× storage saving. Reconstruction:

```
W̃[:, k] = (packed - 8) × scale
```

Maximum representable quantization error per element is `0.5 × scale ≈ max_col / 14`, or roughly **7% relative error per channel** — acceptable for inference when combined with calibration.

---

## Project structure

```
triton_int4/
  __init__.py   — public API surface
  quant.py      — CPU-compatible quantize / dequantize helpers
  kernel.py     — Triton GPU kernel (requires CUDA + Triton)
tests/
  test_quant.py — 26 unit tests (pure CPU, no GPU required)
quant_matmul.py — original prototype (kept for reference)
```

---

## Quick start

```python
import torch
from triton_int4 import quantize_int4, dequantize_int4

# Simulate a weight matrix (e.g. a linear layer)
W = torch.randn(4096, 4096, dtype=torch.float16)

# Compress to INT4
packed, scales = quantize_int4(W)          # packed: uint8 (4096, 2048)
print(f"Original : {W.numel() * 2 / 1e6:.1f} MB (float16)")
print(f"Packed   : {packed.numel() / 1e6:.1f} MB (INT4)   → 4× smaller")

# Reconstruct
W_approx = dequantize_int4(packed, scales)
max_err = (W - W_approx).abs().max().item()
print(f"Max absolute error: {max_err:.4f}")
```

### Fused matmul on GPU (requires Triton)

```python
from triton_int4 import quant_matmul

# Packed weight on CUDA
W = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
packed, scales = quantize_int4(W)
packed, scales = packed.cuda(), scales.cuda()

# Activation
X = torch.randn(32, 4096, dtype=torch.float16, device='cuda')  # batch=32

# Dequantize + matmul inside the kernel — weight never materialized as float16
out = quant_matmul(packed, scales, X)   # (4096, 32)
```

---

## Per-group quantization

For higher accuracy, quantize in *groups* along the K dimension (similar to GPTQ / AWQ):

```python
packed, scales = quantize_int4(W, group_size=128)
# scales shape: (K // group_size,)  — more fine-grained
```

---

## Running the tests

```bash
pip install torch pytest
pytest tests/ -v
```

No GPU or Triton installation required for the test suite.

---

## Why this matters for production LLM serving

- **vLLM / TGI** use INT4/INT8 quantization to pack more model capacity into GPU memory.
- **GPTQ**, **AWQ**, and **bitsandbytes** all rely on variants of the per-channel / per-group INT4 scheme implemented here.
- Fused dequant-matmul (this kernel's approach) avoids materializing the full float16 weight, keeping register pressure low and memory bandwidth minimal.
- Understanding INT4 packing at the Triton level is essential for contributing to or debugging open-source inference engines.

---

## References

- [GPTQ: Accurate Post-Training Quantization for GPT](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [bitsandbytes library](https://github.com/TimDettmers/bitsandbytes)
