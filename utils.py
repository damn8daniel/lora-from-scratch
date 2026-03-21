"""
Utility functions for LoRA/QLoRA framework.
"""

import os
import random
import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """Count total or trainable parameters."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """Print and return trainable vs total parameter counts."""
    trainable = count_parameters(model, only_trainable=True)
    total = count_parameters(model, only_trainable=False)
    ratio = 100.0 * trainable / total if total > 0 else 0.0
    print(f"Trainable: {trainable:,} | Total: {total:,} | Ratio: {ratio:.4f}%")
    return {"trainable": trainable, "total": total, "ratio": ratio}


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    info = {"cpu_rss_mb": 0.0}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        info["cpu_rss_mb"] = process.memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    if torch.cuda.is_available():
        info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    return info


def print_memory_usage(label: str = "") -> None:
    """Print current memory usage."""
    info = get_memory_usage()
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}CPU RSS: {info['cpu_rss_mb']:.1f} MB", end="")
    if "gpu_allocated_mb" in info:
        print(
            f" | GPU Alloc: {info['gpu_allocated_mb']:.1f} MB"
            f" | GPU Reserved: {info['gpu_reserved_mb']:.1f} MB"
            f" | GPU Peak: {info['gpu_max_allocated_mb']:.1f} MB",
            end="",
        )
    print()


@contextmanager
def timer(label: str = ""):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Elapsed: {elapsed:.3f}s")


def find_modules_by_name(model: nn.Module, target_names: list) -> Dict[str, nn.Module]:
    """
    Find all modules whose name contains any of the target substrings.
    Returns dict of {full_name: module}.
    """
    found = {}
    for name, module in model.named_modules():
        for target in target_names:
            if target in name:
                found[name] = module
                break
    return found


def find_linear_modules(model: nn.Module) -> Dict[str, nn.Linear]:
    """Find all nn.Linear modules in a model."""
    return {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}


def find_conv2d_modules(model: nn.Module) -> Dict[str, nn.Conv2d]:
    """Find all nn.Conv2d modules in a model."""
    return {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Conv2d)}


def compute_model_size_mb(model: nn.Module) -> float:
    """Compute model size in MB based on parameter dtypes."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 * 1024)


def estimate_lora_savings(
    model: nn.Module,
    rank: int,
    target_module_count: int,
    avg_dim: Tuple[int, int] = (768, 768),
) -> Dict[str, float]:
    """Estimate memory savings from LoRA vs full fine-tuning."""
    full_ft_params = count_parameters(model)
    # Each LoRA pair: A is (rank x in_dim), B is (out_dim x rank)
    lora_params_per_module = rank * (avg_dim[0] + avg_dim[1])
    total_lora_params = lora_params_per_module * target_module_count

    return {
        "full_ft_params": full_ft_params,
        "lora_params": total_lora_params,
        "reduction_factor": full_ft_params / total_lora_params if total_lora_params > 0 else float("inf"),
        "lora_percent": 100.0 * total_lora_params / full_ft_params if full_ft_params > 0 else 0.0,
    }


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
