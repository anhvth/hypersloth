from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    data=DataConfig(
        dataset_name_or_path='/data-4090/anhvth5/shared-mnt/data/localization/sharegpt_game_631k_messsages.json',
        group_by_length=True,
        # instruction_part="<|im_start|>user\n",
        # response_part="<|im_start|>assistant\n",
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    ),
    training=TrainingConfig(
        gpus=[0,1,2,3,4,5,6,7],
        loss_type="response_only",  # Choices: ["all", "response_only"], the loss will only be calculated on the response part of the input
        # chat_template='google/gemma-3-12b-it'
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-12b-it",
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
    gradient_accumulation_steps=4,  # Meaing 8*4*4=128 examples per step
    num_train_epochs=1,
    learning_rate=1e-4,
    eval_steps=100000,
    logging_steps=1,
    report_to="tensorboard",
    lr_scheduler_type="linear",
    warmup_steps=1,
    save_only_model=True,
    save_steps=200,
    save_total_limit=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    packing=False,
    include_num_input_tokens_seen=True,
    eval_strategy = "epoch",
    # eval_strategy = "steps"
    # eval_steps = 1
)
