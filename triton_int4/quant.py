"""
INT4 symmetric quantization helpers.

Concepts
--------
INT4 has a representable range of [-8, 7] (16 levels).  For each *column* of a
weight matrix we compute a per-channel *scale* so that:

    scale[k]  = max(|W[:, k]|) / 7.0
    q[m, k]   = round(W[m, k] / scale[k])    ∈ [-8, 7]
    packed    = q + 8                         ∈ [0, 15]  (nibble)

Two adjacent columns are packed into a single uint8 byte (nibble packing),
giving the 4× memory reduction relative to float16:

    float16 → 2 bytes/element
    INT4    → 0.5 bytes/element  (4× smaller)

Maximum representable error per element: |W[m,k] - W̃[m,k]| ≤ 0.5 × scale[k]
This is ≤ max|W[:, k]| / 14, i.e. ≤ ~7% of the column dynamic range.

Per-group quantization
----------------------
For finer-grained precision, ``group_size`` splits each *row* into contiguous
groups of columns, each with its own scale:

    scale[m, g]  = max(|W[m, g*gs:(g+1)*gs]|) / 7.0
    scales shape : (M, K // group_size)

This mirrors how GPTQ / AWQ handle quantization.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def quantize_int4(
    weight: torch.Tensor,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float16 matrix to packed INT4 with symmetric scales.

    Parameters
    ----------
    weight:
        2-D float16 tensor of shape ``(M, K)``.  Typically a weight matrix
        where K is the input feature dimension.
    group_size:
        If given, scales are computed per row-group of *group_size* elements
        along K (shape ``(M, K // group_size)``).  ``None`` (default) means
        one scale per column (shape ``(K,)``).

    Returns
    -------
    packed : torch.Tensor
        uint8 tensor of shape ``(M, K // 2)``.  Two consecutive INT4 nibbles
        are packed low-nibble-first into each byte:
          byte[m, j] = (high_nibble << 4) | low_nibble
          where low_nibble  → element (m, 2*j)
                high_nibble → element (m, 2*j + 1)
    scales : torch.Tensor
        float16 tensor of shape ``(K,)`` (per-column) or
        ``(M, K // group_size)`` (per-row-group).

    Raises
    ------
    ValueError
        If ``weight`` is not 2-D, not float16, or K is not even.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    if weight.dtype != torch.float16:
        raise ValueError(f"weight must be float16, got {weight.dtype}")
    M, K = weight.shape
    if K % 2 != 0:
        raise ValueError(f"K={K} must be even for nibble packing")

    # --- compute quantized values -----------------------------------------
    if group_size is None:
        # Per-column: one scale per K index, shared across all M rows
        scales = weight.abs().max(dim=0).values / 7.0           # (K,)
        scales = scales.clamp(min=1e-8)
        scale_broadcast = scales.unsqueeze(0)                   # (1, K)
    else:
        if K % group_size != 0:
            raise ValueError(f"K={K} must be divisible by group_size={group_size}")
        n_groups = K // group_size
        # Per-row-group: each row has its own scale per group
        w_grouped = weight.reshape(M, n_groups, group_size)     # (M, G, gs)
        scales = w_grouped.abs().amax(dim=-1).clamp(min=1e-8) / 7.0  # (M, G)
        # Expand scales to (M, K) for element-wise division
        scale_broadcast = scales.repeat_interleave(group_size, dim=1)  # (M, K)

    q = (weight / scale_broadcast).round().clamp(-8, 7)

    # --- nibble packing ---------------------------------------------------
    # Shift from signed [-8,7] → unsigned [0,15]
    unsigned = (q + 8).to(torch.uint8)                          # (M, K)
    low  = unsigned[:, 0::2]                                    # (M, K//2) even cols
    high = unsigned[:, 1::2]                                    # (M, K//2) odd  cols
    packed = (high << 4) | low                                  # (M, K//2)

    return packed, scales.to(torch.float16)


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a float16 matrix from packed INT4 nibbles and scales.

    Parameters
    ----------
    packed:
        uint8 tensor of shape ``(M, K // 2)`` as returned by
        :func:`quantize_int4`.
    scales:
        float16 tensor of shape ``(K,)`` (per-column) or
        ``(M, K // group_size)`` (per-row-group).

    Returns
    -------
    weight_approx : torch.Tensor
        Reconstructed float16 tensor of shape ``(M, K)``.
    """
    M, half_K = packed.shape
    K = half_K * 2

    # --- unpack nibbles ---------------------------------------------------
    low  = (packed & 0x0F).to(torch.int16)                     # (M, K//2)
    high = ((packed >> 4) & 0x0F).to(torch.int16)              # (M, K//2)

    # Interleave back to (M, K)
    unsigned = torch.empty(M, K, dtype=torch.int16, device=packed.device)
    unsigned[:, 0::2] = low
    unsigned[:, 1::2] = high

    # Dequantize: shift back to signed, then multiply by scale
    signed = (unsigned - 8).to(torch.float16)                  # (M, K)

    if scales.ndim == 1:
        # Per-column scales: (K,)
        if scales.shape[0] != K:
            raise ValueError(
                f"1-D scales must have length K={K}, got {scales.shape[0]}"
            )
        return signed * scales.unsqueeze(0)

    else:
        # Per-row-group scales: (M, G) where G = K // group_size
        n_groups = scales.shape[1]
        group_size = K // n_groups
        scale_broadcast = scales.repeat_interleave(group_size, dim=1)  # (M, K)
        return signed * scale_broadcast
