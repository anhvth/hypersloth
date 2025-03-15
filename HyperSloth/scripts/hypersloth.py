import argparse
from fastcore.all import threaded
from loguru import logger
from transformers.training_args import TrainingArguments


def run(
    gpu: int,
    hyper_config,
    train_args: TrainingArguments,
):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from HyperSloth.transformer_trainer_setup import setup_model_and_training
    from HyperSloth.mmap_gradient_sync import MmapGradSyncCallback


    trainer = setup_model_and_training(
        gpu=gpu,
        hyper_config=hyper_config,
        hf_train_args=TrainingArguments(**train_args),
    )

    if len(hyper_config.training.gpus) > 1:
        grad_sync_cb = MmapGradSyncCallback(
            model=trainer.model,
            grad_dir=hyper_config.grad_dir,
            gpu=gpu,
            gpus=hyper_config.training.gpus,
        )
        logger.info(f"Using gradient sync callback for GPU {gpu}")
        trainer.add_callback(grad_sync_cb)

    trainer.train()


run_in_process = threaded(process=True)(run)

import importlib.util


def load_config_from_path(config_path: str):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def train(config_file: str, **kwargs):
    import os

    config_file = os.path.abspath(config_file)
    assert os.path.exists(config_file), f"Config file {config_file} not found"

    config_module = load_config_from_path(config_file)
    import tabulate
    from fastcore.all import dict2obj, obj2dict

    training_config = config_module.training_config
    hyper_config = dict2obj( config_module.hyper_config)
    _s = {**config_module.hyper_config, **training_config}
    _s = tabulate.tabulate(_s.items(), headers=["Key", "Value"])
    logger.info("\n" + _s)

    logger.info("Cleaning up previous runs")
    os.system(f"rm -rf {hyper_config.grad_dir}/*")

    if len(hyper_config.training.gpus) > 1:
        for gpu_index in hyper_config.training.gpus:
            logger.debug(f"Running on GPU {gpu_index}")
            run_in_process(
                gpu_index,
                hyper_config=hyper_config,
                train_args=training_config,
            )

    else:
        run(
            gpu=hyper_config.training.gpus[0],
            hyper_config=hyper_config,
            train_args=training_config,
        )


def init_config():
    import os

    file = "https://raw.githubusercontent.com/anhvth/hypersloth/refs/heads/main/configs/hypersloth_config_example.py"
    local_file = "hypersloth_config.py"
    os.system(f"wget {file} -O {local_file}")
    logger.info(f"Downloaded {file} to {local_file}")


def main():
    parser = argparse.ArgumentParser(description="HyperSloth CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("config_file", type=str, help="Path to the config file")

    init_parser = subparsers.add_parser("init", help="Initialize the configuration")

    args = parser.parse_args()

    if args.command == "train":
        train(args.config_file)
    elif args.command == "init":
        init_config()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
