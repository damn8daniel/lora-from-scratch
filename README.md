# LoRA from Scratch

A complete from-scratch PyTorch implementation of LoRA (Low-Rank Adaptation) and QLoRA (4-bit Quantized LoRA) for parameter-efficient fine-tuning of neural networks.

## What is LoRA?

Instead of fine-tuning all parameters of a pretrained model, LoRA freezes the base weights and injects trainable low-rank decomposition matrices. For a weight matrix W, the update is:

```
W' = W + (alpha/r) * B @ A
```

where A is (r x d_in), B is (d_out x r), and r << min(d_in, d_out). This reduces trainable parameters by 100-1000x while matching full fine-tuning performance.

## Features

| Feature | Description |
|---------|-------------|
| **LoRALinear** | Low-rank adaptation for nn.Linear with merge/unmerge |
| **LoRAConv2d** | Low-rank adaptation for nn.Conv2d |
| **LoRAEmbedding** | Low-rank adaptation for nn.Embedding |
| **QLoRALinear** | 4-bit NF4 quantized base weights + full-precision LoRA |
| **NF4 Quantization** | NormalFloat4 quantization with per-block scaling |
| **Double Quantization** | Quantize the quantization constants for extra savings |
| **Model Injection** | Automatic LoRA/QLoRA injection into any PyTorch model |
| **Merge/Unmerge** | Fold LoRA weights into base for zero-overhead inference |
| **Multi-adapter** | Configuration support for multiple named adapters |

## Project Structure

```
lora-from-scratch/
├── lora.py          # LoRALinear, LoRAConv2d, LoRAEmbedding
├── qlora.py         # QLoRALinear, NF4 quantization, double quantization
├── inject.py        # Model injection, freeze, merge/unmerge utilities
├── config.py        # LoRAConfig, QLoRAConfig, TrainingConfig, PEFTConfig
├── utils.py         # Seed, parameter counting, memory profiling, device
├── main.py          # Demos: basics, NF4, injection, QLoRA, training
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<user>/lora-from-scratch.git
cd lora-from-scratch
pip install -r requirements.txt
```

## Usage

### Run all demos

```bash
python main.py --demo all
```

This runs five demonstrations:
1. **Basics** — LoRA layer construction, zero-init verification, merge/unmerge
2. **NF4** — 4-bit NormalFloat quantization accuracy and compression ratio
3. **Injection** — LoRA injection into a Transformer with parameter summary
4. **QLoRA** — 4-bit quantized injection with memory savings
5. **Training** — LoRA fine-tuning loop on synthetic next-token prediction

### Run specific demos

```bash
python main.py --demo basics     # LoRA layer mechanics
python main.py --demo nf4        # Quantization accuracy
python main.py --demo injection  # Model injection
python main.py --demo training   # Fine-tuning loop
```

### Use in your own code

```python
import torch.nn as nn
from lora import LoRALinear
from config import LoRAConfig
from inject import inject_lora, freeze_base_model, merge_lora

# Option 1: Direct layer replacement
base_linear = nn.Linear(768, 768)
lora_linear = LoRALinear.from_linear(base_linear, rank=8, alpha=16.0)

# Option 2: Automatic model injection
model = YourModel()
config = LoRAConfig(rank=8, alpha=16.0, target_modules=["query", "value"])
inject_lora(model, config)
freeze_base_model(model)

# Train only LoRA parameters...

# Merge for inference (zero overhead)
merge_lora(model)
```

### QLoRA (4-bit quantization)

```python
from config import QLoRAConfig
from inject import inject_qlora, freeze_base_model

model = YourLargeModel()
config = QLoRAConfig(
    rank=8, alpha=16.0,
    target_modules=["query", "value"],
    blocksize=64,
    double_quant=True,
)
inject_qlora(model, config)
freeze_base_model(model)
# Base weights: 4-bit NF4, LoRA adapters: full precision
```

## How It Works

### LoRA Forward Pass
```
output = base_linear(x) + scaling * (x @ A^T @ B^T)
```
- Base weight W is frozen (no gradients)
- Only A and B receive gradients
- A is initialized with Kaiming uniform, B with zeros (delta_W starts at 0)

### NF4 Quantization
- Weights are divided into blocks of 64 elements
- Each block is normalized by its absolute maximum
- Normalized values are mapped to 16 NormalFloat levels (optimal for Gaussian-distributed weights)
- 4 bits per weight = 8x compression vs FP32
- Double quantization further compresses the per-block scaling factors

### Merge/Unmerge
- **Merge**: `W = W + scaling * B @ A` — fold LoRA into base for inference
- **Unmerge**: `W = W - scaling * B @ A` — restore for continued training
- Merged model has zero inference overhead compared to the original

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022)
