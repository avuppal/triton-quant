# Contributing to triton-quant

Thank you for helping improve triton-quant. This document covers how to add
new quantization kernels, extend to additional bit-widths, and navigate the
inherent tension between accuracy and throughput in quantized inference.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Layout](#project-layout)
3. [Authoring a Quantization Kernel](#authoring-a-quantization-kernel)
4. [Adding a New Bit-Width](#adding-a-new-bit-width)
5. [Accuracy vs. Performance Tradeoffs](#accuracy-vs-performance-tradeoffs)
6. [Testing Conventions](#testing-conventions)
7. [Benchmarking](#benchmarking)
8. [Pull Request Checklist](#pull-request-checklist)

---

## Development Setup

```bash
git clone https://github.com/avuppal/triton-quant.git
cd triton-quant

# CPU-only (sufficient for writing and testing quantization math)
pip install torch pytest

# GPU path (required for Triton kernel work)
pip install torch triton pynvml  # CUDA 11.8+ driver recommended
```

Run the test suite to confirm your environment:

```bash
pytest tests/ -v
```

All 26 tests pass without a GPU. Kernel benchmarks require an NVIDIA GPU.

---

## Project Layout

```
triton_int4/
  __init__.py    # Public API: quantize_int4, dequantize_int4, quant_matmul
  quant.py       # CPU-compatible quantize/dequantize helpers (pure PyTorch)
  kernel.py      # Triton GPU kernel: fused dequant + matmul
tests/
  test_quant.py  # 26 unit tests (CPU only, no GPU required)
quant_matmul.py  # Reference prototype (do not modify; kept for comparison)
```

New bit-widths should be added as separate Python modules under `triton_int4/`
(e.g., `triton_int4/quant_int8.py`, `triton_int4/quant_nf4.py`).

---

## Authoring a Quantization Kernel

### Step 1 — Implement CPU helpers first

The CPU path in `quant.py` is ground-truth. Write `quantize_<fmt>` and
`dequantize_<fmt>` functions in pure PyTorch before touching Triton. This lets
you:

- Validate quantization math against `torch.allclose`
- Achieve 100% test coverage without GPU hardware
- Catch packing bugs (off-by-one nibble shifts, sign errors) cheaply

Required function signatures:

```python
def quantize_<fmt>(
    weight: torch.Tensor,          # float16, 2-D (M, K)
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns (packed, scales)
    ...

def dequantize_<fmt>(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    # Returns float16 approximation
    ...
```

### Step 2 — Write the Triton kernel

The Triton kernel lives in `kernel.py`. The existing INT4 kernel follows this
pattern:

```python
@triton.jit
def _dequant_matmul_kernel(
    packed_ptr, scales_ptr, x_ptr, out_ptr,
    M, N, K,
    stride_pm, stride_pn,       # packed strides
    stride_xm, stride_xk,       # activation strides
    stride_om, stride_on,        # output strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1. Load packed INT4 tile (uint8, half width in K)
    # 2. Unpack nibbles → int16 → float16
    # 3. Multiply by scales
    # 4. Dot-product with activation tile
    # 5. Accumulate and store
```

Key constraints:
- Tile sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) must be powers of two ≥ 16.
- Load `packed` at `K//2` width, then bit-shift to reconstruct both nibbles
  within the same warp — avoids two separate memory transactions.
- Scales should be broadcast along the M dimension (per-column) or via
  `tl.load` with a stride (per-group).
- Use `tl.dot` for the inner-product accumulation to get tensor-core
  acceleration on Ampere+ GPUs.

### Step 3 — Tune autotuning configs

Triton's `@triton.autotune` decorator selects tile configurations. Provide a
realistic spread that covers both small (inference, batch=1) and large (training,
batch=256+) shapes:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 16,  "BLOCK_K": 64}),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 64}),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}),
    ],
    key=["M", "N", "K"],
)
```

---

## Adding a New Bit-Width

### INT8

INT8 has 256 levels ([-128, 127]), so no nibble-packing is needed. Each element
occupies one `uint8` byte — 2× saving vs. float16, not 4×.

| Attribute | INT4 | INT8 |
|-----------|------|------|
| Levels    | 16   | 256  |
| Bytes/elem| 0.5  | 1    |
| vs FP16   | 4×   | 2×   |
| Max quant error | ~7% | ~0.4% |

Implementation notes:
- No nibble-packing loop — `weight.round().clamp(-128, 127).to(torch.int8)`
- Scales: `max(|W|) / 127.0`
- Triton: load INT8 tile directly, convert to float16 with `tl.cast`, then dot.

### NF4 (NormalFloat4)

NF4 (used in QLoRA) maps values to a fixed non-uniform codebook of 16 floats
optimized for normally-distributed weights. Each value is still 4 bits but
the mapping is non-linear.

Implementation notes:
- Define the 16-entry NF4 codebook as a `torch.Tensor` constant.
- Quantize by nearest-neighbor lookup: `torch.argmin(|W_scaled - codebook|)`.
- Pack the 4-bit codebook index identically to INT4 (nibble packing).
- Dequantize: `codebook[index] × scale`.
- The Triton kernel must load the codebook as a `tl.constexpr` or shared memory
  constant — do not recompute it per thread.

### FP8 (E4M3 / E5M2)

NVIDIA H100 supports FP8 natively via NVTE. Triton support is experimental.

- Use `torch.float8_e4m3fn` (PyTorch 2.1+) for the CPU path.
- The Triton kernel can load FP8 tiles and use `tl.dot` directly on H100
  (check `triton.language.dot` type constraints for your Triton version).
- No custom packing needed — FP8 is a standard dtype.

### Checklist for any new bit-width

- [ ] `triton_int4/quant_<fmt>.py` with `quantize_<fmt>` and `dequantize_<fmt>`
- [ ] Exported in `triton_int4/__init__.py`
- [ ] Tests in `tests/test_<fmt>.py` covering:
  - Round-trip identity (quantize → dequantize ≈ original)
  - Per-column and per-group-size scale variants
  - Edge cases: all-zeros, max float16, negative-only tensors
- [ ] Benchmark entry in `README.md` performance table
- [ ] CHANGELOG entry under `[Unreleased]`

---

## Accuracy vs. Performance Tradeoffs

This is the central tension in quantization work. Every design choice moves
the needle on one axis.

### Granularity of scales

| Granularity | Scales tensor shape | Accuracy | Memory overhead | Kernel complexity |
|-------------|---------------------|----------|-----------------|-------------------|
| Per-tensor  | scalar              | Lowest   | Negligible      | Trivial           |
| Per-column  | `(K,)`              | Medium   | ~0.2% of weight | Low               |
| Per-group   | `(M, K//gs)`        | High     | Up to 3%        | Medium            |
| Per-element | `(M, K)`            | Exact    | 100% overhead   | Not useful        |

**Recommendation**: per-column is the right default for a general-purpose
kernel. Per-group (group_size=128) is worth it for large language model weights
(≥1B parameters) where accuracy matters more than the scale overhead.

### Symmetric vs. asymmetric quantization

- **Symmetric** (this repo): `q = round(W / scale)`, zero-point is 0.
  One multiply to dequantize; hardware-friendly.
- **Asymmetric**: `q = round((W - zero_point) / scale)`.
  Better accuracy for activations with non-zero mean (ReLU outputs).
  Adds one addition per element — small but measurable kernel overhead.
  Required for accurate INT4 activation quantization.

### Calibration

Post-training quantization (PTQ) accuracy improves significantly with a
calibration dataset:

1. Run 512–2048 representative samples through the model.
2. Collect per-column activation statistics (`max`, `percentile-99.9`).
3. Use collected stats to set scales instead of per-batch max.

Without calibration, outlier weights inflate the scale and crush small values.
This is the key insight behind GPTQ's second-order optimization and AWQ's
activation-aware scaling.

### Kernel tile size vs. occupancy

Larger tiles (`BLOCK_M=128, BLOCK_N=128`) give better tensor-core utilization
but require more shared memory, reducing SM occupancy. Profile with:

```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active python benchmark.py
```

Target ≥ 60% warp occupancy. If you're below that, reduce tile sizes or
reduce register usage (avoid keeping multiple large tiles live simultaneously).

---

## Testing Conventions

All tests live in `tests/`. Run without a GPU:

```bash
pytest tests/ -v --tb=short
```

**Test categories:**

| Category | What to test |
|----------|-------------|
| **Correctness** | Round-trip quantize→dequantize within expected error bound |
| **Shapes** | Square (4096×4096), rectangular (M≫K, K≫M), minimum (2×2) |
| **Dtypes** | Ensure `float16` input; raise `ValueError` for `float32` |
| **Packing** | Verify nibble packing/unpacking is bit-exact |
| **Scales** | Both per-column and per-group variants |
| **Edge cases** | All-zero weight, single-element columns, K=2 minimum |

Use `torch.allclose(W, W_approx, atol=expected_error_bound)` for round-trip
checks, not exact equality.

---

## Benchmarking

Benchmarks require a CUDA GPU. Run against the existing kernel:

```bash
python -c "
import torch
from triton_int4 import quantize_int4, quant_matmul
import time

M, K, N = 4096, 4096, 4096
W = torch.randn(K, N, dtype=torch.float16, device='cuda')
X = torch.randn(M, K, dtype=torch.float16, device='cuda')
packed, scales = quantize_int4(W)
packed, scales = packed.cuda(), scales.cuda()

# Warm-up
for _ in range(10): quant_matmul(packed, scales, X)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(100): quant_matmul(packed, scales, X)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) * 10  # per iteration in ms
print(f'INT4 fused matmul: {ms:.2f} ms ({2*M*K*N / ms / 1e12:.1f} TFLOP/s)')
"
```

When adding a new kernel, include its numbers in the `README.md` benchmark
table and describe the hardware used (GPU model, driver, CUDA version).

---

## Pull Request Checklist

- [ ] New/modified functions have docstrings
- [ ] Tests added or updated; `pytest tests/ -v` passes locally
- [ ] If GPU kernel: autotuning configs included and tested on ≥1 GPU SKU
- [ ] Benchmark numbers updated in README if performance changes
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] Commit message format: `type(scope): short description`
  - Types: `feat`, `fix`, `perf`, `test`, `docs`, `refactor`
  - Example: `feat(int8): add symmetric INT8 quant helpers and Triton kernel`
