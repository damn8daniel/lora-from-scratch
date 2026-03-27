"""
QLoRA (Quantized LoRA) — 4-bit NormalFloat quantization with LoRA adapters.

Reference: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

Key ideas:
1. Quantize base weights to 4-bit NormalFloat (NF4)
2. Apply LoRA adapters in full precision on top of dequantized weights
3. Optional double quantization of the quantization constants
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora import LoRALayer


# ---------------------------------------------------------------------------
# NF4 quantization
# ---------------------------------------------------------------------------

def compute_nf4_levels() -> torch.Tensor:
    """Compute the 16 NormalFloat4 quantization levels.

    NF4 levels are derived from the quantiles of a standard normal distribution,
    providing optimal information-theoretic representation for normally-distributed
    weights.

    Returns:
        Tensor of 16 quantization levels.
    """
    # Pre-computed NF4 levels (quantiles of N(0,1) for 4-bit representation)
    # Negative levels: 8 quantiles for negative half
    # Positive levels: 7 quantiles for positive half + exact zero
    nf4_levels = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ])
    return nf4_levels


def quantize_nf4(
    weight: torch.Tensor,
    blocksize: int = 64,
) -> tuple:
    """Quantize a weight tensor to 4-bit NormalFloat.

    Args:
        weight: Float weight tensor to quantize.
        blocksize: Number of elements per quantization block.

    Returns:
        quant_indices: Uint8 tensor of quantization indices (packed 2 per byte).
        absmax: Per-block absolute maximum for dequantization.
        shape: Original weight shape.
    """
    shape = weight.shape
    weight_flat = weight.reshape(-1).float()
    n = weight_flat.numel()

    # Pad to blocksize multiple
    if n % blocksize != 0:
        pad = blocksize - (n % blocksize)
        weight_flat = F.pad(weight_flat, (0, pad))
    else:
        pad = 0

    n_padded = weight_flat.numel()
    n_blocks = n_padded // blocksize

    # Reshape into blocks
    blocks = weight_flat.view(n_blocks, blocksize)

    # Per-block absmax normalization
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-10)
    normalized = blocks / absmax.unsqueeze(1)

    # Quantize: find nearest NF4 level
    nf4 = compute_nf4_levels().to(weight.device)
    # Expand for broadcasting: (n_blocks * blocksize, 1) vs (1, 16)
    diffs = (normalized.reshape(-1, 1) - nf4.unsqueeze(0)).abs()
    indices = diffs.argmin(dim=1).to(torch.uint8)

    return indices, absmax, shape, pad


def dequantize_nf4(
    quant_indices: torch.Tensor,
    absmax: torch.Tensor,
    shape: tuple,
    pad: int,
    blocksize: int = 64,
) -> torch.Tensor:
    """Dequantize NF4-quantized weights back to float.

    Args:
        quant_indices: Uint8 quantization indices.
        absmax: Per-block absolute maximums.
        shape: Original weight shape.
        pad: Number of padding elements added.
        blocksize: Block size used during quantization.

    Returns:
        Dequantized weight tensor.
    """
    nf4 = compute_nf4_levels().to(absmax.device)
    n_blocks = absmax.numel()

    # Look up NF4 values
    values = nf4[quant_indices.long()]
    values = values.view(n_blocks, blocksize)

    # Scale by absmax
    dequant = values * absmax.unsqueeze(1)
    dequant = dequant.reshape(-1)

    # Remove padding
    if pad > 0:
        dequant = dequant[:-pad]

    return dequant.reshape(shape)


def double_quantize(
    absmax: torch.Tensor,
    blocksize: int = 256,
) -> tuple:
    """Double quantization: quantize the absmax values to FP8-like format.

    This reduces memory overhead of quantization constants from FP32 to ~FP8.

    Args:
        absmax: FP32 per-block absmax values.
        blocksize: Block size for second-level quantization.

    Returns:
        quant_absmax: Quantized absmax (uint8).
        absmax_absmax: Second-level scaling factors.
        absmax_shape: Original absmax shape.
    """
    shape = absmax.shape
    absmax_flat = absmax.reshape(-1)
    n = absmax_flat.numel()

    if n % blocksize != 0:
        pad = blocksize - (n % blocksize)
        absmax_flat = F.pad(absmax_flat, (0, pad))
    else:
        pad = 0

    blocks = absmax_flat.view(-1, blocksize)
    absmax_absmax = blocks.abs().max(dim=1).values.clamp(min=1e-10)
    normalized = blocks / absmax_absmax.unsqueeze(1)

    # Quantize to 256 levels (uint8)
    quant = (normalized * 127.0).round().clamp(-128, 127).to(torch.int8)

    return quant, absmax_absmax, shape, pad


# ---------------------------------------------------------------------------
# QLoRA Linear layer
# ---------------------------------------------------------------------------

class QLoRALinear(nn.Module, LoRALayer):
    """Linear layer with 4-bit NF4 quantized base weights and LoRA adapters.

    The base weight is stored in quantized form (NF4) and dequantized
    on-the-fly during forward pass. LoRA adapters operate in full precision.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout rate.
        bias: Whether to use bias.
        blocksize: NF4 quantization block size.
        double_quant: Whether to double-quantize absmax values.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        bias: bool = True,
        blocksize: int = 64,
        double_quant: bool = True,
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, rank, alpha, dropout, merge_weights=False)

        self.in_features = in_features
        self.out_features = out_features
        self.blocksize = blocksize
        self.double_quant = double_quant

        # Placeholder for quantized base weight (set via from_linear)
        self.register_buffer("quant_indices", torch.zeros(1, dtype=torch.uint8))
        self.register_buffer("absmax", torch.zeros(1))
        self._weight_shape = (out_features, in_features)
        self._pad = 0

        # Double quantization buffers
        if double_quant:
            self.register_buffer("dq_absmax", None)
            self.register_buffer("dq_absmax_absmax", None)
            self._dq_shape = None
            self._dq_pad = 0

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.bias.requires_grad = False
        else:
            self.register_parameter("bias", None)

        # LoRA adapters (full precision, trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.reset_lora_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        blocksize: int = 64,
        double_quant: bool = True,
    ) -> "QLoRALinear":
        """Create QLoRALinear from a pretrained nn.Linear, quantizing its weights."""
        qlora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
            blocksize=blocksize,
            double_quant=double_quant,
        )

        # Quantize base weight
        indices, absmax, shape, pad = quantize_nf4(linear.weight.data, blocksize)
        qlora.quant_indices = indices
        qlora.absmax = absmax
        qlora._weight_shape = shape
        qlora._pad = pad

        # Double quantize absmax
        if double_quant:
            dq, dq_absmax, dq_shape, dq_pad = double_quantize(absmax)
            qlora.dq_absmax = dq
            qlora.dq_absmax_absmax = dq_absmax
            qlora._dq_shape = dq_shape
            qlora._dq_pad = dq_pad

        # Copy bias
        if linear.bias is not None:
            qlora.bias.data.copy_(linear.bias.data)

        return qlora

    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize the base weight on-the-fly."""
        absmax = self.absmax
        if self.double_quant and self.dq_absmax is not None:
            # Reconstruct absmax from double quantization
            absmax = self.dq_absmax.float() / 127.0
            absmax = absmax.view(-1, absmax.shape[-1] if absmax.ndim > 1 else absmax.numel())
            absmax = absmax * self.dq_absmax_absmax.unsqueeze(1)
            absmax = absmax.reshape(-1)
            if self._dq_pad > 0:
                absmax = absmax[:-self._dq_pad]
            absmax = absmax.reshape(self.absmax.shape)

        return dequantize_nf4(
            self.quant_indices, absmax, self._weight_shape,
            self._pad, self.blocksize,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: dequantize base weight + add LoRA contribution."""
        # Base path (dequantized)
        weight = self._dequantize_weight().to(x.dtype)
        base_out = F.linear(x, weight, self.bias)

        # LoRA path
        lora_out = self.lora_dropout(x)
        lora_out = lora_out @ self.lora_A.T
        lora_out = lora_out @ self.lora_B.T
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    def memory_savings(self) -> dict:
        """Estimate memory savings from quantization."""
        fp32_bytes = self.in_features * self.out_features * 4
        # NF4: 4 bits per weight + absmax overhead
        n_elements = self.in_features * self.out_features
        quant_bytes = n_elements // 2  # 4 bits = 0.5 bytes per element
        n_blocks = math.ceil(n_elements / self.blocksize)
        absmax_bytes = n_blocks * 4  # FP32 absmax
        if self.double_quant:
            absmax_bytes = n_blocks * 1 + math.ceil(n_blocks / 256) * 4

        # LoRA: A (rank x in) + B (out x rank), FP32
        lora_bytes = (self.rank * self.in_features + self.out_features * self.rank) * 4

        total_quant = quant_bytes + absmax_bytes + lora_bytes

        return {
            "fp32_bytes": fp32_bytes,
            "quantized_bytes": total_quant,
            "compression_ratio": fp32_bytes / total_quant if total_quant > 0 else 0,
            "savings_percent": (1 - total_quant / fp32_bytes) * 100 if fp32_bytes > 0 else 0,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"blocksize={self.blocksize}, double_quant={self.double_quant}"
        )
