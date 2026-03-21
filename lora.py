"""
LoRA (Low-Rank Adaptation) layers implemented from scratch.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
https://arxiv.org/abs/2106.09685

Core idea: For a pretrained weight matrix W0 in R^{d x k}, constrain its update
via a low-rank decomposition: W0 + delta_W = W0 + B @ A, where B in R^{d x r},
A in R^{r x k}, and r << min(d, k).

During training, W0 is frozen and only A, B receive gradient updates.
The forward pass computes: h = W0 @ x + (alpha/r) * B @ A @ x
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    """
    Mixin providing LoRA attributes. Not a nn.Module itself; combined with
    Linear/Conv2d via multiple inheritance.
    """

    def __init__(
        self,
        rank: int,
        alpha: float,
        dropout: float,
        merge_weights: bool,
    ):
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False

        if dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=dropout)
        else:
            self.lora_dropout = nn.Identity()

    def reset_lora_parameters(self):
        """Kaiming-uniform for A, zero-init for B — ensures delta_W starts at zero."""
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if hasattr(self, "lora_B"):
            nn.init.zeros_(self.lora_B)


class LoRALinear(nn.Linear, LoRALayer):
    """
    nn.Linear with a parallel low-rank branch.

    Shapes:
        lora_A: (rank, in_features)
        lora_B: (out_features, rank)
        delta_W = lora_B @ lora_A   -> (out_features, in_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        merge_weights: bool = False,
        fan_in_fan_out: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        LoRALayer.__init__(self, rank, alpha, dropout, merge_weights)

        self.fan_in_fan_out = fan_in_fan_out

        # Low-rank factors
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # Freeze the pretrained weight
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.reset_lora_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        merge_weights: bool = False,
    ) -> "LoRALinear":
        """Create a LoRALinear from an existing nn.Linear, preserving weights."""
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights,
            bias=linear.bias is not None,
        )
        lora_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora_linear.bias.data.copy_(linear.bias.data)
        return lora_linear

    def merge(self) -> None:
        """Merge LoRA weights into the base weight for inference."""
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            if self.fan_in_fan_out:
                delta_w = delta_w.T
            self.weight.data += delta_w
            self.merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA weights from the base weight (restore training mode)."""
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            if self.fan_in_fan_out:
                delta_w = delta_w.T
            self.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # LoRA is baked in — standard linear
            if self.fan_in_fan_out:
                return F.linear(x, self.weight.T, self.bias)
            return F.linear(x, self.weight, self.bias)

        # Standard forward
        if self.fan_in_fan_out:
            base_out = F.linear(x, self.weight.T, self.bias)
        else:
            base_out = F.linear(x, self.weight, self.bias)

        # LoRA path: x -> dropout -> A^T -> B^T -> scale
        lora_out = self.lora_dropout(x)
        lora_out = lora_out @ self.lora_A.T  # (batch, seq, rank)
        lora_out = lora_out @ self.lora_B.T  # (batch, seq, out)
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"{base}, rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    nn.Conv2d with a parallel low-rank branch.

    For Conv2d with kernel size (kH, kW):
        Effective weight shape: (out_channels, in_channels * kH * kW)
        lora_A: (rank, in_channels * kH * kW)
        lora_B: (out_channels, rank)

    The LoRA path uses 1x1 convolutions to implement the low-rank factorization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        merge_weights: bool = False,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, rank, alpha, dropout, merge_weights)

        # LoRA factors as 1x1 convolutions for efficient computation
        # A: projects (in_channels * kH * kW) -> rank via grouped 1x1 conv
        # B: projects rank -> out_channels via 1x1 conv
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_channels, *self.kernel_size)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_channels, rank, 1, 1)
        )

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.reset_lora_parameters()

    @classmethod
    def from_conv2d(
        cls,
        conv: nn.Conv2d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        merge_weights: bool = False,
    ) -> "LoRAConv2d":
        """Create LoRAConv2d from existing nn.Conv2d, preserving weights."""
        ks = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        lora_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=ks,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        lora_conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            lora_conv.bias.data.copy_(conv.bias.data)
        return lora_conv

    def merge(self) -> None:
        if not self.merged:
            # lora_B @ lora_A in conv-weight space
            # lora_A: (rank, C_in, kH, kW), lora_B: (C_out, rank, 1, 1)
            # Reshape for matmul: B_2d (C_out, rank) @ A_2d (rank, C_in*kH*kW)
            a_2d = self.lora_A.view(self.rank, -1)            # (r, C_in*kH*kW)
            b_2d = self.lora_B.view(self.weight.size(0), self.rank)  # (C_out, r)
            delta_w = (b_2d @ a_2d).view_as(self.weight) * self.scaling
            self.weight.data += delta_w
            self.merged = True

    def unmerge(self) -> None:
        if self.merged:
            a_2d = self.lora_A.view(self.rank, -1)
            b_2d = self.lora_B.view(self.weight.size(0), self.rank)
            delta_w = (b_2d @ a_2d).view_as(self.weight) * self.scaling
            self.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.conv2d(
                x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            )

        # Base convolution
        base_out = F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

        # LoRA path: apply A conv then B conv (1x1)
        lora_x = self.lora_dropout(x)
        lora_out = F.conv2d(
            lora_x, self.lora_A, None,
            self.stride, self.padding, self.dilation, self.groups,
        )
        lora_out = F.conv2d(lora_out, self.lora_B)
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"{base}, rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    nn.Embedding with LoRA.
    Useful for adapting token embeddings without full retraining.

    lora_A: (rank, num_embeddings)
    lora_B: (embedding_dim, rank)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, rank, alpha, dropout, merge_weights)

        self.lora_A = nn.Parameter(torch.empty(rank, num_embeddings))
        self.lora_B = nn.Parameter(torch.empty(embedding_dim, rank))

        self.weight.requires_grad = False
        self.reset_lora_parameters()

    def merge(self) -> None:
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A).T * self.scaling  # (V, D)
            self.weight.data += delta_w
            self.merged = True

    def unmerge(self) -> None:
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A).T * self.scaling
            self.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.embedding(
            x, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )
        if self.merged:
            return base_out

        # LoRA path via one-hot-like indexing through A
        after_a = F.embedding(
            x, self.lora_A.T, None, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )  # (..., rank)
        lora_out = after_a @ self.lora_B.T  # (..., embedding_dim)
        return base_out + lora_out * self.scaling
