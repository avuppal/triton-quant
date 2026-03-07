# Changelog

All notable changes to triton-quant are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- NF4 (NormalFloat4) codebook quantization — non-uniform 4-bit mapping for
  normally-distributed weights, compatible with QLoRA-style adapters.
- FP8 (E4M3/E5M2) path for H100 — leverages native FP8 GEMM units.
- Asymmetric INT4 (with zero-point) for activation quantization.
- `torch.compile` compatibility layer for the CPU quantization helpers.

---

## [0.3.0] — 2025-01-15

### Added
- **Per-group quantization** (`group_size` parameter in `quantize_int4` and
  `dequantize_int4`). Groups are applied along the K dimension of each row,
  matching the GPTQ/AWQ convention. Scales tensor shape changes from `(K,)`
  to `(M, K // group_size)`.
- Unit tests covering `group_size` variants (16, 32, 64, 128, 256).
- Benchmark comparison between per-column and per-group-128 on A100 SXM4.

### Changed
- `quantize_int4`: scales tensor is now always returned as `float16` (was
  `float32` in some code paths when input had very small magnitudes).

### Fixed
- Off-by-one in per-group scale broadcast when `K % group_size != 0` — now
  raises `ValueError` clearly instead of silently producing wrong shapes.

---

## [0.2.0] — 2024-11-03

### Added
- **Triton GPU kernel** (`triton_int4/kernel.py`): fused dequantize + matmul
  (`quant_matmul`). Dequantizes INT4 weights on-the-fly inside the kernel —
  the float16 weight matrix is never materialized in global memory.
- Autotuning configuration covering BLOCK sizes 16–128 for M/N/K.
- `quant_matmul` exported in `triton_int4/__init__.py`.
- Benchmark results on A100 SXM4 added to README.

### Performance
- INT4 fused kernel achieves ~3.1× throughput improvement vs. float16 `torch.matmul`
  at (4096, 4096, 4096) on A100 SXM4 (memory-bandwidth bound regime).

---

## [0.1.0] — 2024-09-20

### Added
- Initial release of `triton_int4` Python package.
- `quantize_int4`: symmetric per-column INT4 quantization with nibble packing.
  Two INT4 values packed per `uint8` byte (low-nibble-first).
- `dequantize_int4`: reconstruction of float16 matrix from packed nibbles and
  per-column scales.
- 26 unit tests in `tests/test_quant.py` covering correctness, edge cases,
  and error handling. All tests run without GPU or Triton.
- `requirements.txt` with minimal dependencies (`torch`, `triton`).
- `quant_matmul.py` prototype retained as a reference implementation.

---

## References

- [GPTQ: Accurate Post-Training Quantization for GPT](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
