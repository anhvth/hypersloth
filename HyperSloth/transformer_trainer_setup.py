"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os
from loguru import logger

from .hypersloth_config import HyperConfig, TrainingArgsConfig


def setup_model_and_training(
    gpu: int,
    hyper_config: HyperConfig,
    hf_train_args: TrainingArgsConfig,
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.

    Args:
        gpu: GPU index
        hyper_config: Configuration arguments
        hf_train_args: Training arguments

    Returns:
        Trainer object configured for multi-GPU training
    """
    from unsloth import FastModel
    from HyperSloth.dataset_utils import get_chat_dataset
    from trl import SFTTrainer

    gpu_ith = hyper_config.training.gpus[gpu]

    # Initialize model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        **hyper_config.fast_model_args.model_dump()
    )
    if not hyper_config.fast_model_args.full_finetuning:
        model = FastModel.get_peft_model(model, **hyper_config.lora_args.model_dump())

    # Load dataset
    ds_train, ds_test = get_chat_dataset(
        tokenizer=tokenizer, **hyper_config.data.model_dump()
    )

    # Apply PEFT model

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_test if gpu_ith == 0 else None,
        dataset_text_field="text",
        max_seq_length=hyper_config.fast_model_args.max_seq_length,
        dataset_num_proc=hyper_config.data.dataset_num_proc,
        args=hf_train_args,
    )

    # Adjust dataset for multi-GPU training
    max_len_ds = len(hyper_config.training.gpus) * (
        len(trainer.train_dataset) // len(hyper_config.training.gpus)
    )
    trainer.train_dataset = trainer.train_dataset.select(range(max_len_ds))
    trainer.train_dataset = trainer.train_dataset.shard(
        num_shards=len(hyper_config.training.gpus), index=gpu_ith
    )

    # Handle specific training loss type
    if hyper_config.training.loss_type == "response_only":
        from unsloth.chat_templates import train_on_responses_only

        first_text = ds_train[0]["text"]
        instruction_part = hyper_config.data.instruction_part
        response_part = hyper_config.data.response_part
        assert instruction_part in first_text, f"{instruction_part} not in {first_text}"
        assert response_part in first_text, f"{response_part} not in {first_text}"
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
    if gpu_ith == 0:
        logger.info(f"Model setup complete for GPU {gpu_ith}")
        _debug_dataloader(trainer)
    return trainer


def _debug_dataloader(trainer, n_example=10):
    """
    Debug function to log samples from the training dataloader in an HTML format.
    Outputs to both terminal (with colors) and an HTML file with CSS styling.
    """
    from copy import deepcopy

    tokenizer = deepcopy(trainer.tokenizer)
    dl = trainer.get_train_dataloader()
    g = iter(dl)
    html_path = ".log/dataloader_examples.html"
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    
    # Create HTML file with CSS styling
    with open(html_path, "w") as html_file:
        html_file.write("""<!DOCTYPE html>
    <html>
    <head>
        <title>Dataloader Examples</title>
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        
        @media (prefers-color-scheme: light) {
            body { background-color: #ffffff; color: #333; }
            .trainable { background-color: #FFEBCD; color: #333; }
            .context { background-color: #E0FFE0; color: #333; }
            th { background-color: #f2f2f2; }
            th, td { border-color: #ddd; }
        }
        
        @media (prefers-color-scheme: dark) {
            body { background-color: #222; color: #f0f0f0; }
            .trainable { background-color: #664a20; color: #f0f0f0; }
            .context { background-color: #2a5a2a; color: #f0f0f0; }
            th { background-color: #444; color: #f0f0f0; }
            th, td { border-color: #555; }
        }
        
        .trainable, .context { padding: 2px; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid; padding: 8px; text-align: left; }
        h2 { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>Dataloader Examples</h1>
        <p>This file contains examples of training data with context and trainable parts.</p>
    """)
        
        for i in range(n_example):
            batch = next(g)
            input_ids = batch["input_ids"][0]
            label_ids = batch["labels"][0]
            parts_mask = (label_ids >= 0)  # True is trainable, False is context

            # Find split points where trainable/non-trainable sections change
            split_points = [0] + [
                i for i, val in enumerate(parts_mask)
                if i > 0 and val != parts_mask[i - 1]
            ] + [len(parts_mask)]
            
            colored_parts = []
            html_file.write(f"\n    <h2>Example {i+1}</h2>\n")
            html_file.write("    <table>\n        <tr><th>Text</th><th>Label</th></tr>\n")
            
            for a, b in zip(split_points[:-1], split_points[1:]):
                text = tokenizer.decode(input_ids[a:b])
                is_trainable = parts_mask[a]
                
                # Colored text for terminal
                colored_text = f"\033[93m{text}\033[0m" if is_trainable else f"\033[92m{text}\033[0m"
                colored_parts.append(colored_text)
                
                # HTML with CSS classes
                css_class = "trainable" if is_trainable else "context"
                label = "🟠 TRAIN" if is_trainable else "🟢 CONTEXT"
                
                # Escape HTML special characters
                text_escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                
                # Add row to HTML table
                html_file.write(f"        <tr>\n            <td><span class=\"{css_class}\">{text_escaped}</span></td>\n"
                               f"            <td>{label}</td>\n        </tr>\n")
            
            html_file.write("    </table>\n")
            
            # Colored text for terminal
            colored_output = "".join(colored_parts)
            terminal_msg = f"\n=== EXAMPLE #{i+1} ===\n" + colored_output + "\n"
            if i == 0:
                print(terminal_msg)
        
        html_file.write("</body>\n</html>")
    
    print(f"More training debug examples written to {html_path}")
