from typing import List, Literal, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from multiprocessing import cpu_count

CPU_COUNT = min(1, cpu_count() - 2)


class DataConfig(BaseModel):
    """Configuration for dataset handling and processing."""

    dataset_name_or_path: Union[str, list] = "data/cod_1k.json"
    dataset_num_proc: int = CPU_COUNT
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"
    num_samples: Optional[int] = None
    group_by_length: bool = True
    split: str = "train"

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class TrainingConfig(BaseModel):
    """Configuration for training setup and parameters."""

    gpus: List[int] = Field(default=[0], description="List of GPU indices to use")
    loss_type: Literal["all", "response_only"] = Field(
        default="response_only",
        description="Loss calculation type: 'all' or 'response_only'",
    )
    chat_template: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Chat template for formatting input data",
    )

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class FastModelArgs(BaseModel):
    """Configuration for Unsloth's FastModel initialization."""

    model_name: str = "unsloth/gemma-3-4b-it"
    max_seq_length: Optional[int] = None
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class LoraArgs(BaseModel):
    """Configuration for LoRA parameters in PEFT."""

    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class HyperConfig(BaseModel):
    """Main configuration class combining all sub-configurations."""

    grad_dir: str = "/dev/shm/hypersloth"
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    fast_model_args: FastModelArgs = Field(default_factory=FastModelArgs)
    lora_args: LoraArgs = Field(default_factory=LoraArgs)
    use_mmap_grad_sync: bool = Field(default=True)
    pretrained_lora: Optional[str] = Field(
        default=None,
        description="Path to pretrained LoRA model for continous lora training",
    )

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class TrainingArgsConfig(BaseModel):
    """Configuration for Hugging Face TrainingArguments."""

    output_dir: str = "saves/loras/"
    per_device_train_batch_size: int = 8
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 16
    logging_steps: int = 1
    report_to: str = "tensorboard"
    num_train_epochs: int = 1
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 5
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    save_only_model: bool = True
    seed: int = 42
    save_only_model: bool = True

    eval_strategy: str = "no"  # must be no, when using multigpus
    per_device_eval_batch_size: int = 2
    include_num_input_tokens_seen: bool = True
    include_tokens_per_second: bool = True

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TrainingArguments initialization."""
        return self.model_dump()
