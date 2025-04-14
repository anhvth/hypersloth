import os
import sys
import time
import warnings
import importlib.util
from fastcore.all import threaded, call_parse
from loguru import logger
import tabulate

from HyperSloth.hypersloth_config import HyperConfig, TrainingArgsConfig
from speedy_utils import setup_logger

if not "HYPERSLOTH_CACHE_DIR" in os.environ:
    os.environ["HYPERSLOTH_CACHE_DIR"] = "/dev/shm/hypersloth/"
# os.environ['HF_HOME'] = '/data-4090/huggingace-cache'
warnings.filterwarnings("ignore")
# os.environ["UNSLOTH_ENABLE_LOGGING"] = "0"

def get_run_id(hyper_config_model, training_config_model):
    """
    Generate a descriptive and concise run ID based on the hyperparameters and training configuration.
    """
    model_name = hyper_config_model.fast_model_args.model_name.split("/")[-1]
    if isinstance(hyper_config_model.data.dataset_name_or_path, list):
        dataset = "_".join(
            [path.split("/")[-1].split(".")[0] for path in hyper_config_model.data.dataset_name_or_path]
        )
    else:
        dataset = hyper_config_model.data.dataset_name_or_path.split("/")[-1].split('.')[0]
    loss_type = hyper_config_model.training.loss_type
    lora_r = hyper_config_model.lora_args.r
    lora_alpha = hyper_config_model.lora_args.lora_alpha
    seq_len = hyper_config_model.fast_model_args.max_seq_length
    lr = training_config_model.learning_rate
    batch_size = training_config_model.per_device_train_batch_size
    accum_steps = training_config_model.gradient_accumulation_steps
    ngpu = len(hyper_config_model.training.gpus)
    epochs = training_config_model.num_train_epochs
    seed = training_config_model.seed
    mmap_sync = "mmap" if hyper_config_model.use_mmap_grad_sync else "no-mmap"
    
    # Calculate global batch size
    global_bz = batch_size * accum_steps * ngpu

    run_id = (
        f"loss-{loss_type}_lora-r{lora_r}-a{lora_alpha}_"
        f"seq-{seq_len}_lr-{lr}_global_bz-{global_bz}_"
        f"epochs-{epochs}_seed-{seed}_{mmap_sync}-{ngpu}gpu"
    )
    # normalize remove special characters like .
    run_id = run_id.replace(".", "_").replace("-", "_")

    return f'{model_name}_{dataset}', run_id

def _get_hp_grad_dir(model_name_dataset, run_id):
    grad_dir = os.path.join(os.environ["HYPERSLOTH_CACHE_DIR"], model_name_dataset,f"run_{run_id}")
    os.makedirs(grad_dir, exist_ok=True)
    return grad_dir

def _setup_logger(gpu_id):
    lvl = os.environ.get("HYPERSLOTH_LOG_LEVEL", "D")
    setup_logger(lvl, disable_grep='mmap_gradient_sync')
    file = f".log/process_{gpu_id}.log"
    if os.path.exists(file):
        os.remove(file)

    
    print(f"Logging to {file}")

def _train(gpu: int, hyper_config: HyperConfig, hf_train_args: TrainingArgsConfig):
    _setup_logger(f"{gpu}")
    hf_train_args.output_dir = os.path.join(hf_train_args.output_dir, *get_run_id(hyper_config, hf_train_args))
    logger.info(f"Training on GPU {gpu} with output_dir {hf_train_args.output_dir}")

    from HyperSloth.hp_trainer_setup import setup_model_and_training  # avoid circular import
    os.environ["HYPERSLOTH_LOCAL_RANK"] = str(hyper_config.training.gpus.index(gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    trainer, model, tokenizer = setup_model_and_training(
        hyper_config=hyper_config,
        hf_train_args=hf_train_args,
    )

    if len(hyper_config.training.gpus) > 1 and hyper_config.use_mmap_grad_sync is not None:
        from HyperSloth.mmap_gradient_sync import MmapGradSyncCallback
        if hyper_config.use_mmap_grad_sync:
            grad_sync_cb = MmapGradSyncCallback(
                model=trainer.model,
                grad_dir=os.environ["HYPERSLOTH_RUN_DIR"],
                gpu=gpu,
                gpus=hyper_config.training.gpus,
            )
            logger.info(f"Using gradient sync callback for GPU {gpu}")
            trainer.add_callback(grad_sync_cb)
    else:
        logger.warning("Gradient sync is not enabled, will use normal unsloth training")
    trainer.train()

    # Save once from rank=0
    if gpu == hyper_config.training.gpus[0]:
        logger.info(f"Save model to {hf_train_args.output_dir}")
        model.save_pretrained(hf_train_args.output_dir)
        tokenizer.save_pretrained(hf_train_args.output_dir)



def load_config_from_path(config_path: str):
    """Load configuration from Python file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

# We'll just detect if the user wants a tmux script:


def build_tmux_script(
    session_name: str,
    script_path: str,
    model_name_dataset: str,
    run_id: str,
    config_file: str,
    gpus: list
):
    """
    Build a script that:
    1. Kills any existing tmux session with `session_name`
    2. Creates a new session for the first GPU
    3. Creates new windows for the remaining GPUs
    4. Sends the appropriate commands to each window
    Saves the final script to `script_path`.
    """
    lines = []
    lines.append("#!/usr/bin/env bash")
    # remove grad_dir
    lines.append(f"rm -rf {_get_hp_grad_dir(model_name_dataset, run_id)}")
    lines.append(f"""# Create a new session with first GPU = 0
tmux new-session -d -s {session_name} -n MAIN""")
    

    # First GPU
    # check tmux session command, if yes, ask user enter "y" to kill the session
    # check_if_session_exists_then_ask_to_kill = f"tmux has-session -t {session_name} && read -p 'Session exists, kill it? (y/n): ' kill_session && [ $kill_session == 'y' ] && tmux kill-session -t {session_name}"
    # lines.append(check_if_session_exists_then_ask_to_kill)
    # Remaining GPUs
    for local_rank, gpu_index in enumerate(gpus):
        cmd = (
            f"USE_TMUX=0 "
            f"python {sys.argv[0]} "
            f"{config_file} "
            f"--rank {local_rank} "
            f"--world_size {len(gpus)}"
        )
        lines.append(f"tmux new-window -t {session_name} -n gpu_{gpu_index}")
        lines.append(
            f"tmux send-keys -t {session_name}:gpu_{gpu_index} '{cmd}' Enter"
        )
        lines.append("")

    lines.append(f'echo "Attach to this session via: tmux attach -t {session_name}"')

    # Write out the script
    script_body = "\n".join(lines)
    with open(script_path, "w") as f:
        f.write(script_body)
    os.chmod(script_path, 0o755)
    
    is_session_exists = os.system(f"tmux has-session -t {session_name}")
    if is_session_exists == 0:
        logger.warning(f"Session {session_name} exists, please kill it before running the script")
    else:
        os.system(f"bash {script_path}")
        logger.info(f"Script started with session name {session_name}")
        
        

@call_parse
def train(config_file: str, rank: int = None, world_size: int = None, use_tmux: bool = False):
    
        
    config_file, hyper_config, training_config = initialize_training_config(config_file)
    # clean grad_dir
    # CASE 1: Child process => single GPU
    model_name_dataset, run_id = get_run_id(hyper_config, training_config)
    os.environ["HYPERSLOTH_RUN_DIR"] = _get_hp_grad_dir(model_name_dataset, run_id)
    # _clean_grad_dir(rank, world_size, use_tmux)
    # clearn dir
    if rank is not None and world_size is not None:
        logger.info(f"[CASE 1] Running on rank {rank} with world size {world_size}")
        _train(
            gpu=hyper_config.training.gpus[rank],
            hyper_config=hyper_config,
            hf_train_args=training_config,
        )
        return

    # CASE 2: Top-level process => spawn multi-GPU or single GPU
    gpus = hyper_config.training.gpus

    # If multiple GPUs:
    if len(gpus) > 1:
        if os.environ.get("USE_TMUX", "0") == "1" or use_tmux:
            # Build a tmux script that the user can run manually
            # session_name = f"train_hp_{model_name_dataset}_{run_id}"
            session_name = f"train_hp"
            script_path = "/tmp/hp_train.sh"
            build_tmux_script(session_name, script_path, model_name_dataset, run_id, config_file, gpus)
            return
        else:
            # Launch via multi-processing (no tmux).
            logger.info(f"[CASE 2] Running on {len(gpus)} GPUs")
            processes = []
            assert len(gpus) > 1, "Cannot use multi-processing with a single GPU"
            @threaded(process=True)
            def run_in_process(*args, **kwargs):
                """Runs _train() in a separate Python process."""
                _train(*args, **kwargs)
            for gpu_index in gpus:
                p = run_in_process(
                    gpu_index,
                    hyper_config=hyper_config,
                    hf_train_args=training_config,
                )
                processes.append(p)

            # Wait for processes; if one errors, kill them all
            while processes:
                for proc in processes:
                    if not proc.is_alive():
                        if proc.exitcode != 0:
                            for p in processes:
                                p.terminate()
                            logger.error("Error in training, terminating all processes")
                            raise Exception("Error in training")
                        else:
                            processes.remove(proc)
                            break
                time.sleep(1)
            logger.success("All processes finished")

    else:
        # Single GPU
        assert not use_tmux, "Cannot use tmux with a single GPU"
        _train(
            gpu=gpus[0],
            hyper_config=hyper_config,
            hf_train_args=training_config,
        )

def initialize_training_config(config_file):
    # global USE_TMUX
    # USE_TMUX = USE_TMUX or use_tmux
    """Train entry-point. If rank/world_size are provided, we assume this is 
    a child process that trains on a single GPU. Otherwise, 
    we spawn multi-gpu runs either by generating a tmux script or by multi-process.
    """
    
    config_file = os.path.abspath(config_file)
    assert os.path.exists(config_file), f"Config file {config_file} not found"

    config_module = load_config_from_path(config_file)

    # Retrieve configs from the module
    if hasattr(config_module, "hyper_config_model"):
        hyper_config = config_module.hyper_config_model
    elif hasattr(config_module, "hyper_config"):
        hyper_config = HyperConfig(**config_module.hyper_config)
    else:
        hyper_config = HyperConfig()

    if hasattr(config_module, "training_config_model"):
        training_config = config_module.training_config_model
    elif hasattr(config_module, "training_config"):
        training_config = TrainingArgsConfig(**config_module.training_config)
    else:
        raise ValueError("No training configuration found")

    # Display combined config
    combined_config = {**hyper_config.model_dump(), **training_config.model_dump()}
    config_table = tabulate.tabulate(combined_config.items(), headers=["Key", "Value"])
    logger.info("\n" + config_table)

    # # of GPUs
    os.environ["HYPERSLOTH_NUM_GPUS"] = str(len(hyper_config.training.gpus))
    return config_file,hyper_config,training_config
