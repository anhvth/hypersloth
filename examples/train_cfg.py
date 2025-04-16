from HyperSloth.hypersloth_config import *

# Main configuration using Pydantic models
hyper_config_model = HyperConfig(
    data=DataConfig(
        dataset_name_or_path=[
            "/home/anhvth5/LLaMA-Factory/data/translation_v3_seql_4k_packing.json",
            "/shared-mnt/data/localization/translation_v3_refine_12484_packing_4k.json",
            "/shared-mnt/data/sharegpt/game_multilingal_synthetic_12k_gpt4p1_packing_seql4k.json",
            "/shared-mnt/data/sharegpt/train_ds_spliter_sql4k.json"
        ],
        group_by_length=True,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        num_samples=700,
        test_ratio=0.01,
        shuffle_user_dict_keys=True,
    ),
    training=TrainingConfig(
        gpus=range(8),
        loss_type="response_only",  # Choices: ["all", "response_only"], the loss will only be calculated on the response part of the input
    ),
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-27b-it-bnb-4bit",
        max_seq_length=4096,
    ),
    pretrained_lora="saves/translation_v3_ft_27b_ckpt_400",
    lora_args=LoraArgs(
        r=16,
        lora_alpha=16,
    ),
)

# Training arguments using Pydantic model
training_config_model = TrainingArgsConfig(
    output_dir="/shared-mnt/hypersloth_loras/",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Meaing 8*4*4=128 examples per step
    num_train_epochs=1,
    learning_rate=1e-5,
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
    eval_strategy="epoch",
)
