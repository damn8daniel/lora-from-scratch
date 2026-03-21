"""
Configuration dataclasses for LoRA/QLoRA fine-tuning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    fan_in_fan_out: bool = False
    merge_weights: bool = False
    enable_lora: bool = True

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (Quantized LoRA)."""
    bits: int = 4
    quant_type: str = "nf4"          # "nf4" or "fp4"
    double_quant: bool = True
    double_quant_type: str = "fp8"
    blocksize: int = 64
    paged_optimizer: bool = True
    paged_optimizer_offload_threshold: float = 0.7  # fraction of GPU mem


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 8
    max_seq_length: int = 256
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    output_dir: str = "./output"
    device: str = "cuda"


@dataclass
class AdapterConfig:
    """Configuration for a single named adapter."""
    name: str = "default"
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    active: bool = True


@dataclass
class PEFTConfig:
    """Top-level PEFT configuration supporting multiple adapters."""
    base_model_name: str = ""
    adapters: Dict[str, AdapterConfig] = field(default_factory=dict)
    active_adapter: str = "default"
    inference_mode: bool = False

    def add_adapter(self, name: str, config: LoRAConfig) -> None:
        self.adapters[name] = AdapterConfig(name=name, lora_config=config)

    def set_active(self, name: str) -> None:
        if name not in self.adapters:
            raise KeyError(f"Adapter '{name}' not found. Available: {list(self.adapters.keys())}")
        self.active_adapter = name
