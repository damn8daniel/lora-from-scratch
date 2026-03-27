"""
LoRA from Scratch — Demo & Verification.

Demonstrates LoRA and QLoRA on a small Transformer-like model:
1. LoRA injection and parameter efficiency
2. Forward pass equivalence before/after merge
3. QLoRA quantization and memory savings
4. Training loop on synthetic data
"""

import argparse
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lora import LoRALinear, LoRAConv2d, LoRAEmbedding
from qlora import QLoRALinear, quantize_nf4, dequantize_nf4
from config import LoRAConfig, QLoRAConfig
from inject import (
    inject_lora, inject_qlora, freeze_base_model,
    merge_lora, unmerge_lora, count_lora_parameters,
    print_lora_summary, get_lora_parameters,
)
from utils import set_seed, print_trainable_parameters, print_memory_usage


# ---------------------------------------------------------------------------
# Toy Transformer model
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    """Small Transformer for demonstration."""

    def __init__(self, vocab_size=1000, d_model=256, n_heads=4,
                 n_layers=4, d_ff=512, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_lora_basics():
    """Demonstrate basic LoRA layer functionality."""
    print("=" * 60)
    print("LoRA Layer Basics")
    print("=" * 60)

    # Create base linear and LoRA version
    base = nn.Linear(256, 512)
    lora = LoRALinear.from_linear(base, rank=8, alpha=16.0)

    x = torch.randn(2, 10, 256)

    # Before any training, LoRA output should equal base output
    # (because B is initialized to zero)
    with torch.no_grad():
        base_out = base(x)
        lora_out = lora(x)
        diff = (base_out - lora_out).abs().max().item()
    print(f"\nLoRALinear(256, 512, rank=8):")
    print(f"  Max diff from base (initial): {diff:.2e}")

    # Count parameters
    base_params = sum(p.numel() for p in base.parameters())
    lora_trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    lora_total = sum(p.numel() for p in lora.parameters())
    print(f"  Base params: {base_params:,}")
    print(f"  LoRA trainable: {lora_trainable:,} ({100*lora_trainable/base_params:.2f}% of base)")
    print(f"  LoRA total: {lora_total:,}")

    # Merge/unmerge test
    lora.lora_A.data.normal_(0, 0.1)  # Simulate some training
    with torch.no_grad():
        out_before = lora(x).clone()
        lora.merge()
        out_merged = lora(x).clone()
        lora.unmerge()
        out_unmerged = lora(x).clone()

    print(f"\n  Merge/unmerge roundtrip error: {(out_before - out_unmerged).abs().max().item():.2e}")
    print(f"  Merged output matches: {(out_before - out_merged).abs().max().item():.2e}")

    # Conv2d LoRA
    base_conv = nn.Conv2d(3, 64, 3, padding=1)
    lora_conv = LoRAConv2d.from_conv2d(base_conv, rank=4)
    x_img = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        diff = (base_conv(x_img) - lora_conv(x_img)).abs().max().item()
    print(f"\nLoRAConv2d(3, 64, 3x3, rank=4):")
    print(f"  Max diff from base (initial): {diff:.2e}")


def demo_nf4_quantization():
    """Demonstrate NF4 quantization accuracy."""
    print("\n" + "=" * 60)
    print("NF4 Quantization")
    print("=" * 60)

    # Simulate normally-distributed weights
    weight = torch.randn(512, 256)

    # Quantize and dequantize
    indices, absmax, shape, pad = quantize_nf4(weight, blocksize=64)
    weight_recon = dequantize_nf4(indices, absmax, shape, pad, blocksize=64)

    error = (weight - weight_recon).abs()
    print(f"\nWeight shape: {weight.shape}")
    print(f"  Mean abs error: {error.mean().item():.6f}")
    print(f"  Max abs error: {error.max().item():.6f}")
    print(f"  Relative RMSE: {(error**2).mean().sqrt().item() / weight.abs().mean().item():.4f}")

    # Memory comparison
    fp32_bytes = weight.numel() * 4
    nf4_bytes = weight.numel() // 2 + absmax.numel() * 4
    print(f"\n  FP32 size: {fp32_bytes:,} bytes")
    print(f"  NF4 size: {nf4_bytes:,} bytes")
    print(f"  Compression: {fp32_bytes / nf4_bytes:.1f}x")


def demo_model_injection():
    """Demonstrate LoRA injection into a Transformer."""
    print("\n" + "=" * 60)
    print("LoRA Model Injection")
    print("=" * 60)

    model = ToyTransformer(vocab_size=1000, d_model=256, n_heads=4, n_layers=4)

    print(f"\nBase model:")
    print_trainable_parameters(model)

    # Inject LoRA into attention layers
    config = LoRAConfig(rank=8, alpha=16.0, dropout=0.05,
                        target_modules=["query", "value"])
    inject_lora(model, config)
    freeze_base_model(model)

    print(f"\nAfter LoRA injection (query + value):")
    print_trainable_parameters(model)
    print_lora_summary(model)


def demo_qlora_injection():
    """Demonstrate QLoRA injection."""
    print("\n" + "=" * 60)
    print("QLoRA (4-bit) Model Injection")
    print("=" * 60)

    model = ToyTransformer(vocab_size=1000, d_model=256, n_heads=4, n_layers=4)

    print(f"\nBase model:")
    base_info = print_trainable_parameters(model)

    config = QLoRAConfig(rank=8, alpha=16.0, dropout=0.05,
                         target_modules=["query", "value"],
                         blocksize=64, double_quant=True)
    inject_qlora(model, config)
    freeze_base_model(model)

    print(f"\nAfter QLoRA injection:")
    print_trainable_parameters(model)
    print_lora_summary(model)

    # Check memory savings per layer
    for name, module in model.named_modules():
        if isinstance(module, QLoRALinear):
            savings = module.memory_savings()
            print(f"\n  {name}: {savings['compression_ratio']:.1f}x compression "
                  f"({savings['savings_percent']:.1f}% savings)")
            break  # just show one example


def demo_training():
    """Demonstrate LoRA fine-tuning on a simple task."""
    print("\n" + "=" * 60)
    print("LoRA Fine-Tuning Demo")
    print("=" * 60)

    set_seed(42)
    device = torch.device("cpu")

    # Create model and inject LoRA
    model = ToyTransformer(vocab_size=100, d_model=128, n_heads=4,
                           n_layers=2, d_ff=256, max_len=32).to(device)

    config = LoRAConfig(rank=4, alpha=8.0, dropout=0.05,
                        target_modules=["query", "value", "out_proj"])
    inject_lora(model, config)
    freeze_base_model(model)

    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(lora_params, lr=1e-3)
    counts = count_lora_parameters(model)
    print(f"\nTrainable parameters: {counts['lora']:,} / {counts['total']:,} "
          f"({counts['lora_percent']:.2f}%)")

    # Synthetic next-token prediction task
    batch_size = 8
    seq_len = 16

    print(f"\nTraining on synthetic next-token prediction (batch={batch_size}, seq={seq_len}):")
    model.train()
    t0 = time.perf_counter()

    for step in range(1, 51):
        x = torch.randint(0, 100, (batch_size, seq_len), device=device)
        target = torch.randint(0, 100, (batch_size, seq_len), device=device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 100), target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

    elapsed = time.perf_counter() - t0
    print(f"Training time: {elapsed:.2f}s")

    # Test merge for inference
    model.eval()
    x_test = torch.randint(0, 100, (1, seq_len), device=device)
    with torch.no_grad():
        out_before = model(x_test).clone()
        merge_lora(model)
        out_merged = model(x_test).clone()

    diff = (out_before - out_merged).abs().max().item()
    print(f"\nMerge verification (max diff): {diff:.2e}")
    print("Merge successful!" if diff < 1e-4 else "WARNING: merge mismatch!")


def main():
    parser = argparse.ArgumentParser(description="LoRA from Scratch Demo")
    parser.add_argument("--demo", type=str, default="all",
                        choices=["all", "basics", "nf4", "injection",
                                 "qlora", "training"],
                        help="Which demo to run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.demo in ("all", "basics"):
        demo_lora_basics()

    if args.demo in ("all", "nf4"):
        demo_nf4_quantization()

    if args.demo in ("all", "injection"):
        demo_model_injection()

    if args.demo in ("all", "qlora"):
        demo_qlora_injection()

    if args.demo in ("all", "training"):
        demo_training()

    print("\nAll demos completed successfully.")


if __name__ == "__main__":
    main()
