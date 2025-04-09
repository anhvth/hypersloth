"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os

from HyperSloth._utils import (configure_batch_size, create_trainer,
                               init_model_and_tokenizer)

SLEEP_WAIT_DATASET_TIMEOUT = 1800
from .hypersloth_config import HyperConfig, TrainingArgsConfig


def setup_model_and_training(
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.
    """
    gpu_ith = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
    num_gpus = int(os.environ["HYPERSLOTH_NUM_GPUS"])
    configure_batch_size(hf_train_args, gpu_ith, num_gpus)
    model, tokenizer = init_model_and_tokenizer(hyper_config)
    trainer = create_trainer(tokenizer, hyper_config, hf_train_args, gpu_ith, model)
    return trainer, model, tokenizer
