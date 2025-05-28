<p align="center">
    <img src="images/hpsloth.webp" alt="HyperSloth Logo" width="200" />
</p>

# Hyper-Sloth

A high-performance framework for fine-tuning large language models.

## Performance Benchmarks

HyperSloth demonstrates significant performance improvements over other popular fine-tuning frameworks, powered by NCCL's optimized communication backend.

### Training Time Comparison (4x RTX 4090)

| Framework           | Training Time | VRAM Peak Consumption | Communication Backend |
| ------------------- | ------------- | --------------------- | --------------------- |
| HyperSloth (NCCL)   | 19 minutes    | 6 GB                  | NCCL                  |
| HyperSloth (Legacy) | ~21 minutes   | 6 GB                  | Memory-mapped files   |
| LlamaFactory        | 30 minutes    | 21 GB                 | PyTorch DDP           |
| Unsloth (1X)        | ~70 minutes   | 6 GB                  | Single GPU            |

## Overview

HyperSloth is an extension of Unsloth for distributed training of Large Language Models across multiple GPUs. It uses NCCL as the backbone for efficient gradient synchronization, providing production-ready distributed training capabilities.

## Features

- **NCCL-based gradient synchronization**: High-performance distributed training using industry-standard NCCL backend
- **Efficient weight synchronization**: Ensure model consistency across all GPUs during training
- **Template fixes**: Custom tokenizer chat template fixes for proper handling of "think" tags
- **Customizable loss types**: Support for full sequence or response-only training
- **Educational memory-mapped approach**: Includes deprecated memory-mapped gradient sync for learning purposes

## Installation

```bash
# Clone the repository
pip install git+https://github.com/anhvth/HyperSloth.git
```

## Architecture

### Current Implementation (Production)

HyperSloth now uses **NCCL (NVIDIA Collective Communications Library)** as its backbone for distributed training. This provides:

- **Industry-standard performance**: Leverages NVIDIA's optimized communication primitives
- **Robust fault tolerance**: Built-in error handling and retry mechanisms
- **Scalable communication**: Efficient all-reduce operations across multiple GPUs
- **Production stability**: Battle-tested in enterprise environments

### Legacy Implementation (Educational)

The original memory-mapped file approach (`/dev/shm`) is still included for educational purposes, demonstrating:

- **Custom gradient synchronization**: Manual coordination using memory-mapped files
- **Lock-based coordination**: File locking mechanisms for process synchronization
- **Low-level distributed concepts**: Understanding the fundamentals of gradient aggregation

> **Note**: The memory-mapped implementation is deprecated and will be removed in future versions. It serves as a learning resource for understanding distributed training internals.

### Train a model across multiple GPUs

```bash
# Create a config file for training
[>training| ~/projects/hyper-sloth ] hypersloth-init
# Example training config: ./hs_training_config.py
hypersloth-train ./hs_training_config.py

# [>training| ~/projects/hyper-sloth ] hypersloth-train ./hs_training_config.py
# 2025-03-16 06:53:56.861 | INFO     | HyperSloth.scripts.trainner:train:94 -
# Key                          Value
# ---------------------------  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# grad_dir                     /dev/shm/hypersloth  # Note: Only used for legacy mode, NCCL uses native PyTorch distributed
# data                         {'dataset_name_or_path': 'mlabonne/FineTome-100k', 'test_ratio': 0.05, 'dataset_num_proc': 4, 'instruction_part': '<start_of_turn>user\n', 'response_part': '<start_of_turn>model\n', 'num_samples': 1000, 'split': 'train'}
# training                     {'gpus': [0, 1], 'loss_type': 'response_only', 'packing': False}
# fast_model_args              {'model_name': 'unsloth/gemma-3-1b-it', 'max_seq_length': 2048, 'load_in_4bit': True, 'load_in_8bit': False, 'full_finetuning': False, 'token': None}
# lora_args                    {'finetune_vision_layers': False, 'finetune_language_layers': True, 'finetune_attention_modules': True, 'finetune_mlp_modules': True, 'r': 16, 'lora_alpha': 16, 'lora_dropout': 0.0, 'bias': 'none', 'random_state': 3407}
# output_dir                   outputs/2B/
# per_device_train_batch_size  4
# learning_rate                0.0002
# gradient_accumulation_steps  16
# per_device_eval_batch_size   2
# eval_steps                   100
# logging_steps                1
# report_to                    tensorboard
# num_train_epochs             1
# lr_scheduler_type            linear
# warmup_steps                 5
# seed                         42
# save_total_limit             2
# bf16                         True
# fp16                         False
# optim                        adamw_8bit
# weight_decay                 0.01
# packing                      False
# 2025-03-16 06:53:56.861 | INFO     | HyperSloth.scripts.trainner:train:97 - Cleaning up previous runs
# 2025-03-16 06:53:56.868 | DEBUG    | HyperSloth.scripts.trainner:train:103 - Running on GPU 0 (NCCL backend)
# 2025-03-16 06:53:57.870 | DEBUG    | HyperSloth.scripts.trainner:train:103 - Running on GPU 1 (NCCL backend)
```

> **Note**: The current implementation automatically uses NCCL for gradient synchronization. The `grad_dir` parameter is maintained for backward compatibility but is not used in the NCCL implementation.

### Loss Curves

The loss convergence between HyperSloth and LlamaFactory is comparable, indicating similar training quality with significantly improved training speed and reduced memory consumption.

| ![Hyper-Sloth Tensorboard](images/hyper-sloth-tb.png){ width=150 } | ![LlamaFactory Tensorboard](images/llama-factory-tb.png){ width=150 } |
| ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Hyper-Sloth Tensorboard[^1]                                        | LlamaFactory Tensorboard[^2]                                          |

[^1]: Hyper-Sloth Tensorboard.
[^2]: LlamaFactory Tensorboard.
