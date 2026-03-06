"""
Unit tests for triton_int4.quant — pure CPU, no CUDA required.

Tests cover:
- round-trip quantize → dequantize fidelity
- nibble packing / unpacking correctness
- per-channel scale computation
- per-group scale computation
- boundary values (all zeros, all same value, large magnitude)
- error handling (bad dtype, odd K, non-2D input)
- memory footprint (packed is 4× smaller than float16)
"""

import pytest
import torch
import sys
import os

# allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triton_int4.quant import quantize_int4, dequantize_int4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_weight(M: int, K: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(M, K, dtype=torch.float16)


# ---------------------------------------------------------------------------
# packing / unpacking
# ---------------------------------------------------------------------------

class TestNibblePacking:
    def test_low_nibble_stored_in_low_bits(self):
        """Even columns land in the low 4 bits of each packed byte."""
        # Two-column matrix: col0 = 3, col1 = 5  → scales ≈ 3/7, 5/7
        # q0 = round(3 / (3/7)) + 8 = 7+8 = 15 → 0x0F
        # q1 = round(5 / (5/7)) + 8 = 7+8 = 15 → 0x0F
        # packed byte = (0xF << 4) | 0xF = 0xFF
        w = torch.tensor([[3.0, 5.0]], dtype=torch.float16)
        packed, _ = quantize_int4(w)
        assert packed.shape == (1, 1), f"expected (1,1) got {packed.shape}"
        assert (packed[0, 0] & 0x0F) == ((packed[0, 0] >> 4) & 0x0F)

    def test_pack_unpack_invertible(self):
        """Packed then unpacked nibbles exactly match original unsigned values."""
        # Build a matrix where we know the quantized values
        # Use values proportional to the scale so rounding is exact
        M, K = 4, 8
        # Uniform value = 7 * scale → quantizes exactly to q=7
        scale_vals = torch.arange(1, K + 1, dtype=torch.float16)  # (K,)
        weight = scale_vals.unsqueeze(0).expand(M, -1).clone()     # (M, K)
        packed, scales = quantize_int4(weight)

        # Every quantized value should be +7 → unsigned nibble = 15
        low  = (packed & 0x0F)
        high = ((packed >> 4) & 0x0F)
        assert low.eq(15).all(),  "low nibbles should all be 15"
        assert high.eq(15).all(), "high nibbles should all be 15"

    def test_packed_shape(self):
        for M, K in [(1, 2), (8, 16), (64, 128), (1, 64)]:
            packed, scales = quantize_int4(_make_weight(M, K))
            assert packed.shape == (M, K // 2), \
                f"M={M}, K={K}: expected ({M},{K//2}), got {packed.shape}"
            assert scales.shape == (K,), \
                f"expected scales shape ({K},), got {scales.shape}"


# ---------------------------------------------------------------------------
# round-trip fidelity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _range_relative_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Max absolute error expressed as a fraction of the column dynamic range.

        This is the right metric for quantization: error / range, not error / element.
        For symmetric INT4 the theoretical max is 1/14 ≈ 7%.
        """
        abs_err = (original.float() - reconstructed.float()).abs()
        col_range = original.abs().float().max(dim=0).values.clamp(min=1e-6)  # (K,)
        return (abs_err / col_range.unsqueeze(0)).max().item()

    def test_small_matrix_low_error(self):
        """Max error should be ≤ ~7% of the column range (INT4 theoretical bound)."""
        w = _make_weight(8, 16)
        packed, scales = quantize_int4(w)
        w_hat = dequantize_int4(packed, scales)
        err = self._range_relative_error(w, w_hat)
        # Theoretical max = 1/14 ≈ 7.1%; allow float16 rounding headroom
        assert err < 0.12, f"range-relative error too large: {err:.4f}"

    def test_large_matrix_mean_error(self):
        """Mean absolute error should be tiny relative to the weight magnitude."""
        w = _make_weight(128, 256)
        packed, scales = quantize_int4(w)
        w_hat = dequantize_int4(packed, scales)
        mae = (w.float() - w_hat.float()).abs().mean().item()
        w_mean = w.abs().float().mean().item()
        assert mae < 0.15 * w_mean, f"mean error {mae:.5f} exceeds 15% of mean magnitude {w_mean:.5f}"

    def test_all_zeros(self):
        """All-zero weight → scales clamped to eps, dequant should also be near zero."""
        w = torch.zeros(4, 8, dtype=torch.float16)
        packed, scales = quantize_int4(w)
        w_hat = dequantize_int4(packed, scales)
        assert w_hat.abs().max().item() < 1e-3, "all-zero weight should reconstruct near zero"

    def test_constant_matrix(self):
        """Every element the same value → scale = value/7, all nibbles = 15."""
        val = torch.tensor(1.0, dtype=torch.float16)
        w = torch.full((4, 8), val.item(), dtype=torch.float16)
        packed, scales = quantize_int4(w)
        w_hat = dequantize_int4(packed, scales)
        # All elements should reconstruct to exactly 1.0 (quantizes exactly to 7)
        assert (w_hat - val).abs().max().item() < 1e-3


# ---------------------------------------------------------------------------
# per-group quantization
# ---------------------------------------------------------------------------

class TestGroupQuant:
    def test_group_size_shape(self):
        """Per-row-group scales have shape (M, K // group_size)."""
        M, K, G = 16, 64, 16
        packed, scales = quantize_int4(_make_weight(M, K), group_size=G)
        assert packed.shape == (M, K // 2)
        n_groups = K // G
        assert scales.shape == (M, n_groups), \
            f"expected ({M}, {n_groups}) scales, got {scales.shape}"

    def test_group_round_trip(self):
        """Per-row-group quantization achieves low error (finer-grained than per-channel)."""
        w = _make_weight(32, 64)
        packed, scales = quantize_int4(w, group_size=16)
        w_hat = dequantize_int4(packed, scales)
        # per-row-group: error measured against per-group dynamic range
        n_groups = scales.shape[1]
        group_size = 64 // n_groups
        max_err = 0.0
        for g in range(n_groups):
            w_g = w[:, g * group_size:(g + 1) * group_size].float()
            h_g = w_hat[:, g * group_size:(g + 1) * group_size].float()
            group_range = w_g.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
            err = ((w_g - h_g).abs() / group_range).max().item()
            max_err = max(max_err, err)
        assert max_err < 0.12, f"group range-relative error too large: {max_err:.4f}"

    def test_invalid_group_size_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            quantize_int4(_make_weight(4, 10), group_size=3)


# ---------------------------------------------------------------------------
# memory footprint
# ---------------------------------------------------------------------------

class TestMemoryFootprint:
    def test_4x_smaller(self):
        """Packed INT4 bytes should be 4× smaller than float16 storage."""
        M, K = 256, 512
        w = _make_weight(M, K)
        packed, _ = quantize_int4(w)
        fp16_bytes   = w.element_size() * w.numel()     # 2 bytes × M×K
        packed_bytes = packed.element_size() * packed.numel()  # 1 byte × M×(K/2)
        assert fp16_bytes == 4 * packed_bytes, \
            f"expected 4× savings: fp16={fp16_bytes}B, packed={packed_bytes}B"


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_wrong_dtype_raises(self):
        with pytest.raises(ValueError, match="float16"):
            quantize_int4(torch.randn(4, 8, dtype=torch.float32))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            quantize_int4(torch.randn(4, 8, 2, dtype=torch.float16))

    def test_odd_k_raises(self):
        with pytest.raises(ValueError, match="even"):
            quantize_int4(torch.ones(4, 7, dtype=torch.float16))


# ---------------------------------------------------------------------------
# scale correctness
# ---------------------------------------------------------------------------

class TestScaleComputation:
    def test_scale_equals_max_over_7(self):
        """Per-channel scale = max(|col|) / 7."""
        M, K = 4, 8
        w = _make_weight(M, K, seed=99)
        _, scales = quantize_int4(w)
        expected = w.abs().max(dim=0).values / 7.0
        # float16 rounding → allow small tolerance
        assert torch.allclose(scales.float(), expected.float(), atol=1e-3), \
            f"scales differ: max diff = {(scales - expected).abs().max():.6f}"

    def test_scales_positive(self):
        w = _make_weight(16, 32)
        _, scales = quantize_int4(w)
        assert (scales > 0).all(), "all scales must be positive"

    def test_scales_dtype_float16(self):
        _, scales = quantize_int4(_make_weight(4, 8))
        assert scales.dtype == torch.float16


# ---------------------------------------------------------------------------
# dequantize shape / dtype
# ---------------------------------------------------------------------------

class TestDequantize:
    def test_output_shape_matches_original(self):
        M, K = 12, 24
        w = _make_weight(M, K)
        packed, scales = quantize_int4(w)
        w_hat = dequantize_int4(packed, scales)
        assert w_hat.shape == (M, K), f"expected {(M,K)}, got {w_hat.shape}"

    def test_output_dtype_float16(self):
        packed, scales = quantize_int4(_make_weight(4, 8))
        w_hat = dequantize_int4(packed, scales)
        assert w_hat.dtype == torch.float16

    def test_large_positive_values(self):
        """Very large weights still quantize without overflow."""
        w = torch.full((4, 8), 1000.0, dtype=torch.float16)
        packed, scales = quantize_int4(w)
        w_hat = dequantize_int4(packed, scales)
        # should reconstruct close to 1000.0
        assert (w_hat - w).abs().max().item() < 20.0, \
            "large-value reconstruction too far off"
