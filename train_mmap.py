import argparse
import os
from functools import partial

from loguru import logger
from speedy_utils.all import multi_thread

from notveryslow.mmap_gradient_sync import MmapGradSyncCallback

multi_thread = partial(multi_thread, report=False, verbose=False)

from transformers import TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script for multi-GPU setup.")
    parser.add_argument(
        "--gpu_index", type=int, default=0, help="Index of the GPU to use."
    )
    parser.add_argument(
        "--visible_devices",
        type=int,
        nargs="+",
        default=[0],
        help="List of visible GPU devices.",
    )
    parser.add_argument(
        "--file", type=str, default="./data/cod_6k5.json", help="Path to the data file."
    )
    parser.add_argument("--packing", action="store_true", help="Enable packing.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.05, help="Ratio of the test set."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_seq_length",
        "-l",
        type=int,
        default=2048,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        "-b",
        type=int,
        default=1,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=1, help="Log every X updates steps."
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Run an evaluation every X steps."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of steps to perform learning rate warmup.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate for Adam.",
    )
    parser.add_argument(
        "--not_bf16",
        dest="bf16",
        action="store_false",
        default=True,
        help="Whether to use 16-bit (mixed) precision training.",
    )
    parser.add_argument(
        "--loss_type",
        choices=["all", "target_only"],
        default="target_only",
        help="Whether to use 16-bit (mixed) precision training.",
    )
    parser.add_argument(
        "--optim", type=str, default="adamw_8bit", help="Optimizer to use."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to use."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--seed", type=int, default=3407, help="Random seed for initialization."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_training_outputs/debug",
        help="The output directory for the model training.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Limit the total amount of checkpoints.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The list of integrations to report the results and logs to.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    from notveryslow.transformer_trainer_setup import setup_model_and_training

    all_gpus = args.visible_devices
    args.is_main = args.gpu_index == args.visible_devices[0]

    train_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.is_main else "no",
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        do_eval=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not args.bf16,
        bf16=args.bf16,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=f"{args.output_dir}/{args.gpu_index}",
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
    )

    trainer = setup_model_and_training(args=args, train_args=train_args)

    grad_sync_cb = MmapGradSyncCallback(
        model=trainer.model,
        grad_dir="./grads",
        gpu_index=args.gpu_index,
        visible_devices=all_gpus,
    )
    if len(args.visible_devices) > 1:
        logger.info(f"Using gradient sync callback for GPU {args.gpu_index}")
        trainer.add_callback(grad_sync_cb)

    trainer.train()


if __name__ == "__main__":
    main()
