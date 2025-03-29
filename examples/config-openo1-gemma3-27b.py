from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    data=DataConfig(
        dataset_name_or_path="/shared-mnt/data/sharegpt/teacher_messages_deepseek.json",
        group_by_length=True,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        # num_samples=1000,
    ),
    training=TrainingConfig(
        gpus=range(8),
        loss_type="response_only",  # Choices: ["all", "response_only"], the loss will only be calculated on the response part of the input
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-27b-it-bnb-4bit",
        max_seq_length=7_000,
    ),
    # pretrained_lora="/shared-mnt/loras/gemma-3-27b-it-bnb-4bit_teacher_messages_deepseek_direct/loss_response_only_lora_r16_a16_seq_7000_lr_0_0001_global_bz_16_epochs_2_seed_42_mmap/",
    lora_args=LoraArgs(
        r=16,
        lora_alpha=16,
    ),
    
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="/shared-mnt/loras/",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Meaing 8*4*4=128 examples per step
    num_train_epochs=2,
    learning_rate=1e-4,
    per_device_eval_batch_size=4,
    eval_steps=100000,
    logging_steps=1,
    report_to="tensorboard",
    lr_scheduler_type="linear",
    warmup_steps=0,
    save_only_model=True,
    save_steps=200,
    save_total_limit=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    packing=False,
    include_num_input_tokens_seen=True,
)
