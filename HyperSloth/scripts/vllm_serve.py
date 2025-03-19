import os
import subprocess
import time
from typing import List, Optional
from fastcore.script import call_parse
from ray import logger


def kill_existing_vllm(vllm_binary: Optional[str] = None) -> None:
    """Kill selected vLLM processes using fzf."""
    if not vllm_binary:
        vllm_binary = get_vllm()

    # List running vLLM processes
    result = subprocess.run(
        f"ps aux | grep {vllm_binary} | grep -v grep",
        shell=True,
        capture_output=True,
        text=True,
    )
    processes = result.stdout.strip().split("\n")

    if not processes or processes == [""]:
        print("No running vLLM processes found.")
        return

    # Use fzf to select processes to kill
    fzf = subprocess.Popen(
        ["fzf", "--multi"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    selected, _ = fzf.communicate("\n".join(processes))

    if not selected:
        print("No processes selected.")
        return

    # Extract PIDs and kill selected processes
    pids = [line.split()[1] for line in selected.strip().split("\n")]
    for pid in pids:
        subprocess.run(
            f"kill -9 {pid}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(f"Killed processes: {', '.join(pids)}")



@call_parse
def main(
    model: str,
    gpu_groups: str,
    served_model_name: Optional[str] = None,
    port_start: int = 8155,
    gpu_memory_utilization: float = 0.95,
    dtype: str = "bfloat16",
    max_model_len: int = 8192,
    enable_lora: bool = False,
    enable_quantization: bool = False,
    not_verbose=True,
    extra_args: Optional[List[str]] = []
):
    """Main function to start or kill vLLM containers."""


    """Start vLLM containers with dynamic args."""
    gpu_groups_arr = gpu_groups.split(",")
    VLLM_BINARY = get_vllm()
    if enable_lora:
        VLLM_BINARY = 'VLLM_ALLOW_RUNTIME_LORA_UPDATING=True ' + VLLM_BINARY

    # Auto-detect quantization based on model name if not explicitly set
    if (
        not enable_quantization
        and model
        and ("bnb" in model.lower() or "4bit" in model.lower())
    ):
        enable_quantization = True
        print(f"Auto-detected quantization for model: {model}")

    # Set environment variables for LoRA if needed
    if enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        print("Enabled runtime LoRA updating")

    for i, gpu_group in enumerate(gpu_groups_arr):
        port = port_start + i
        gpu_group = ",".join([str(x) for x in gpu_group])
        tensor_parallel = len(gpu_group.split(","))

        cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_group}",
            VLLM_BINARY,
            "serve",
            model,
            "--port",
            str(port),
            "--tensor-parallel",
            str(tensor_parallel),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--dtype",
            dtype,
            "--max-model-len",
            str(max_model_len),
            "--disable-log-requests",
        ]
        if not_verbose:
            cmd += ["--uvicorn-log-level critical", "--enable-prefix-caching"]

        if served_model_name:
            cmd.extend(["--served-model-name", served_model_name])

        if enable_quantization:
            cmd.extend(
                ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
            )

        if enable_lora:
            cmd.extend(["--fully-sharded-loras", "--enable-lora"])
        # add kwargs
        if extra_args:
            cmd += extra_args
        final_cmd = " ".join(cmd)
        log_file = f"/tmp/vllm_{port}.txt"
        final_cmd_with_log = f'"{final_cmd} 2>&1 | tee {log_file}"'
        run_in_tmux = (
            f"tmux new-session -d -s vllm_{port} 'bash -c {final_cmd_with_log}'"
        )

        print(final_cmd)
        print("Logging to", log_file)
        os.system(run_in_tmux)



def get_vllm():
    VLLM_BINARY = subprocess.check_output("which vllm", shell=True, text=True).strip()
    VLLM_BINARY = os.getenv("VLLM_BINARY", VLLM_BINARY)
    logger.info(f"vLLM binary: {VLLM_BINARY}")
    assert os.path.exists(
        VLLM_BINARY
    ), f"vLLM binary not found at {VLLM_BINARY}, please set VLLM_BINARY env variable"
    return VLLM_BINARY
