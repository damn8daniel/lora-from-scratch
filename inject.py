"""
LoRA injection utilities — replace standard layers with LoRA-augmented versions.

Provides functions to:
1. Inject LoRA into specific modules of a model
2. Freeze base model parameters
3. Collect trainable (LoRA-only) parameters
4. Merge/unmerge LoRA weights for inference
"""

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from lora import LoRALinear, LoRAConv2d, LoRAEmbedding
from qlora import QLoRALinear
from config import LoRAConfig, QLoRAConfig


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Inject LoRA adapters into target modules of a model.

    Args:
        model: The base model to augment.
        config: LoRA configuration.
        target_modules: List of module name substrings to target.
            If None, uses config.target_modules.

    Returns:
        The model with LoRA layers injected (modified in-place).
    """
    target_modules = target_modules or config.target_modules
    replaced = {}

    for name, module in model.named_modules():
        if not _should_replace(name, target_modules):
            continue

        parent_name, attr_name = _split_name(name)
        parent = _get_module(model, parent_name) if parent_name else model

        if isinstance(module, nn.Linear):
            new_module = LoRALinear.from_linear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                merge_weights=config.merge_weights,
            )
            setattr(parent, attr_name, new_module)
            replaced[name] = "LoRALinear"

        elif isinstance(module, nn.Conv2d):
            new_module = LoRAConv2d.from_conv2d(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                merge_weights=config.merge_weights,
            )
            setattr(parent, attr_name, new_module)
            replaced[name] = "LoRAConv2d"

        elif isinstance(module, nn.Embedding):
            new_module = LoRAEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                rank=config.rank,
                alpha=config.alpha,
            )
            new_module.weight.data.copy_(module.weight.data)
            setattr(parent, attr_name, new_module)
            replaced[name] = "LoRAEmbedding"

    return model


def inject_qlora(
    model: nn.Module,
    config: QLoRAConfig,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Inject QLoRA (4-bit quantized LoRA) into target Linear modules.

    Args:
        model: The base model.
        config: QLoRA configuration.
        target_modules: Module name substrings to target.

    Returns:
        The model with QLoRA layers injected.
    """
    target_modules = target_modules or config.target_modules
    replaced = {}

    for name, module in list(model.named_modules()):
        if not _should_replace(name, target_modules):
            continue

        if not isinstance(module, nn.Linear):
            continue

        parent_name, attr_name = _split_name(name)
        parent = _get_module(model, parent_name) if parent_name else model

        new_module = QLoRALinear.from_linear(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            blocksize=config.blocksize,
            double_quant=config.double_quant,
        )
        setattr(parent, attr_name, new_module)
        replaced[name] = "QLoRALinear"

    return model


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapters."""
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


def unfreeze_lora(model: nn.Module) -> None:
    """Ensure all LoRA parameters are trainable."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return only the LoRA adapter parameters."""
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return state dict containing only LoRA parameters."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def merge_lora(model: nn.Module) -> None:
    """Merge all LoRA weights into base weights for efficient inference."""
    for module in model.modules():
        if hasattr(module, "merge") and callable(module.merge):
            module.merge()


def unmerge_lora(model: nn.Module) -> None:
    """Unmerge LoRA weights from base weights (restore for continued training)."""
    for module in model.modules():
        if hasattr(module, "unmerge") and callable(module.unmerge):
            module.unmerge()


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """Count LoRA vs base parameters."""
    lora_params = 0
    base_params = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_params += param.numel()
        else:
            base_params += param.numel()
    return {
        "lora": lora_params,
        "base": base_params,
        "total": lora_params + base_params,
        "lora_percent": 100.0 * lora_params / (lora_params + base_params)
        if (lora_params + base_params) > 0 else 0.0,
    }


def print_lora_summary(model: nn.Module) -> None:
    """Print a summary of LoRA-injected modules."""
    print("\nLoRA Module Summary:")
    print(f"{'Module':>40} {'Type':>15} {'Rank':>5} {'Params':>10}")
    print("-" * 75)

    total_lora = 0
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d, LoRAEmbedding, QLoRALinear)):
            lora_p = sum(p.numel() for n, p in module.named_parameters() if "lora_" in n)
            total_lora += lora_p
            mtype = type(module).__name__
            rank = module.rank
            print(f"  {name:>38} {mtype:>15} {rank:>5} {lora_p:>10,}")

    counts = count_lora_parameters(model)
    print("-" * 75)
    print(f"  Total LoRA params: {counts['lora']:,} ({counts['lora_percent']:.2f}%)")
    print(f"  Total base params: {counts['base']:,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_replace(name: str, targets: List[str]) -> bool:
    """Check if a module name matches any target substring."""
    return any(t in name for t in targets)


def _split_name(name: str):
    """Split 'a.b.c' into ('a.b', 'c')."""
    parts = name.rsplit(".", 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[1]


def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Get a submodule by dotted name."""
    parts = name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module
