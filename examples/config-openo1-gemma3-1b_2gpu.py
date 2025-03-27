from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    data=DataConfig(
        dataset_name_or_path="data/OpenO1-SFT",
        group_by_length=True,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        num_samples=100,
    ),
    training=TrainingConfig(
        gpus=range(2),
        loss_type="response_only",  # Choices: ["all", "response_only"], the loss will only be calculated on the response part of the input
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-1b-it-bnb-4bit",
        max_seq_length=8_000,
    ),
    lora_args=LoraArgs(
        r=16,
        lora_alpha=16,
    ),
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="saves/loras/",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # Meaing 8*4*4=128 examples per step
    learning_rate=1e-4,
    per_device_eval_batch_size=4,
    eval_steps=100000,
    logging_steps=1,
    report_to="tensorboard",
    num_train_epochs=2,
    lr_scheduler_type="linear",
    warmup_steps=0,
    save_only_model=True,
    save_steps=200,
    save_total_limit=2,

    optim="adamw_8bit",
    weight_decay=0.01,
    packing=False,
    include_num_input_tokens_seen=True,
)
