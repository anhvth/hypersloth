"""
Memory-mapped gradient synchronization for distributed training.

WARNING: This module is deprecated and will be removed in a future version.
Please use the native distributed training capabilities of your framework instead.
"""

import os
import time
import numpy as np
import torch
from filelock import FileLock
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from loguru import logger

TIME_OUT = 1800
SLEEP_TIME = 0.01
WAIT_WARNING_THRESHOLD = 2  # Log a warning if waiting longer than this


class Flag:
    def __init__(self, world_size: int, file_path: str, is_master: bool = False):
        self.world_size = world_size
        self.file_path = file_path
        self.is_master = is_master
        self.lock_path = self.file_path + ".lock"
        self.lock = FileLock(self.lock_path)

        if self.is_master:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(self.file_path, "wb") as f:
                f.truncate(world_size * 4)
        else:
            while not os.path.exists(self.file_path):
                time.sleep(SLEEP_TIME)

        self.mem = np.memmap(
            self.file_path,
            dtype="float32",
            mode="r+",
            shape=(self.world_size,),
        )

        if self.is_master:
            self.reset()

    def update(self, rank: int):
        with self.lock:
            self.mem[rank] = 1.0
            self.mem.flush()

    def wait_for_all(self, step: int, timeout: float = TIME_OUT):
        t0 = time.time()
        has_logged = False

        while True:
            with self.lock:
                flags_copy = self.mem.copy()
            if np.all(flags_copy == 1.0):
                break

            elapsed = time.time() - t0
            if elapsed > WAIT_WARNING_THRESHOLD and not has_logged:
                logger.warning(
                    f"[Flag={self.file_path}] waiting {elapsed:.1f}s at step={step}, flags={flags_copy.tolist()}"
                )
                has_logged = True

            if elapsed > timeout:
                raise RuntimeError(
                    f"[Flag={self.file_path}] Timeout after {elapsed:.1f}s waiting at step={step}, flags={flags_copy.tolist()}"
                )

            time.sleep(SLEEP_TIME)

        logger.debug(f"[Flag={self.file_path}] all ranks ready at step={step}")

    def reset(self):
        if not self.is_master:
            raise RuntimeError("Only master can reset a flag array.")
        with self.lock:
            self.mem[:] = 0.0
            self.mem.flush()


class MmapGradientSync:
    def __init__(
        self,
        model: torch.nn.Module,
        gpu: int,
        gpus: list,
        grad_dir: str = "/dev/shm/hypersloth/",
    ):
        self.gpu = gpu
        self.gpus = gpus
        self.local_rank = gpus.index(gpu)
        self.num_gpus = len(gpus)
        self.is_main = self.local_rank == 0

        os.makedirs(".cache", exist_ok=True)
        log_file = f".cache/gpu_{self.local_rank}.log"
        main_log_file = f".cache/gpu_main.log"

        self.logger = logger.bind()
        # remove existing handlers
        # self.logger.remove()
        self.logger.add(
            log_file, format="{time} {level} {message}", level="DEBUG", enqueue=True
        )
        self.logger.add(
            main_log_file,
            format="{time} {level} {message}",
            level="DEBUG",
            enqueue=True,
        )

        os.makedirs(grad_dir, exist_ok=True)

        self.grad_path = os.path.join(grad_dir, "grad_buffer_v2.dat")
        self.grad_lock = FileLock(self.grad_path + ".lock")

        self.param_info = []
        total_grad_size = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                numel = param.numel()
                shape = param.shape
                self.param_info.append(
                    {
                        "name": name,
                        "offset": total_grad_size,
                        "numel": numel,
                        "shape": shape,
                    }
                )
                total_grad_size += numel * self.num_gpus

        if self.is_main:
            with open(self.grad_path, "wb") as f:
                f.truncate(total_grad_size * 4)
        else:
            while not os.path.exists(self.grad_path):
                time.sleep(SLEEP_TIME)

        self.grad_mem = np.memmap(
            self.grad_path, dtype="float32", mode="r+", shape=(total_grad_size,)
        )

        self.flag_dir = os.path.join(grad_dir, "flags")
        os.makedirs(self.flag_dir, exist_ok=True)

        self.flags = {
            "ready_to_start": Flag(
                self.num_gpus,
                os.path.join(self.flag_dir, "ready_to_start.dat"),
                is_master=self.is_main,
            ),
            "write": Flag(
                self.num_gpus,
                os.path.join(self.flag_dir, "write.dat"),
                is_master=self.is_main,
            ),
            "read": Flag(
                self.num_gpus,
                os.path.join(self.flag_dir, "read.dat"),
                is_master=self.is_main,
            ),
            "iteration_done": Flag(
                self.num_gpus,
                os.path.join(self.flag_dir, "iteration_done.dat"),
                is_master=self.is_main,
            ),
        }

        self.step_file = os.path.join(self.flag_dir, "step.dat")
        self.step_lock = FileLock(self.step_file + ".lock")
        if self.is_main:
            with open(self.step_file, "wb") as f:
                f.truncate(4)
        else:
            while not os.path.exists(self.step_file):
                time.sleep(SLEEP_TIME)

        self.step_mem = np.memmap(
            self.step_file, dtype="float32", mode="r+", shape=(1,)
        )
        if self.is_main:
            self.step_mem[0] = 0.0
            self.step_mem.flush()

    def _get_current_step(self) -> int:
        with self.step_lock:
            return int(self.step_mem[0])

    def _set_current_step(self, step: int):
        assert self.is_main
        with self.step_lock:
            self.step_mem[0] = float(step)
            self.step_mem.flush()

    def _increment_global_step(self):
        new = self._get_current_step() + 1
        self._set_current_step(new)
        self.logger.info(f"[GPU={self.gpu}] Master incremented step to {new}")

    def _check_step_matches_local(self, local_step: int):
        return self._get_current_step() == local_step

    def accumulate_local_grad(self, model: torch.nn.Module, local_step: int):
        self.flags["ready_to_start"].update(self.local_rank)
        if self.is_main:
            self.flags["ready_to_start"].wait_for_all(local_step)
            self.flags["ready_to_start"].reset()

        if not self._check_step_matches_local(local_step):
            raise RuntimeError(
                f"[GPU={self.gpu}] Step mismatch: local={local_step}, global={self._get_current_step()}"
            )

        named_params = dict(model.named_parameters())
        written = 0
        with self.grad_lock:
            for info in self.param_info:
                name = info["name"]
                offset = info["offset"]
                numel = info["numel"]
                param = named_params[name]
                if param.grad is not None:
                    flat = param.grad.detach().cpu().numpy().ravel()
                    start = offset + self.local_rank * numel
                    self.grad_mem[start : start + numel] = flat
                    written += numel
            self.grad_mem.flush()

        self.logger.debug(
            f"[GPU={self.gpu}] Wrote {written} grad floats at step={local_step}"
        )
        self.flags["write"].update(self.local_rank)

    def read_final_grad_into_model(
        self, model: torch.nn.Module, local_step: int, average=True
    ):
        self.flags["write"].wait_for_all(local_step)

        named_params = dict(model.named_parameters())
        with self.grad_lock:
            for info in self.param_info:
                name = info["name"]
                offset = info["offset"]
                numel = info["numel"]
                shape = info["shape"]
                param = named_params[name]
                acc = np.zeros(numel, dtype=np.float32)
                for r in range(self.num_gpus):
                    acc += self.grad_mem[offset + r * numel : offset + (r + 1) * numel]
                if average:
                    acc /= self.num_gpus
                param.grad = torch.from_numpy(acc.reshape(shape)).to(param.device)

        self.flags["read"].update(self.local_rank)

    def zero_mmaps(self, local_step: int):
        self.flags["read"].wait_for_all(local_step)
        if self.is_main:
            with self.grad_lock:
                self.grad_mem[:] = 0.0
                self.grad_mem.flush()
        self.flags["iteration_done"].update(self.local_rank)

    def iteration_done(self, local_step: int):
        if self.is_main:
            self.flags["iteration_done"].wait_for_all(local_step)
            for key in ["write", "read", "iteration_done"]:
                self.flags[key].reset()
            self._increment_global_step()
        else:
            while self._get_current_step() <= local_step:
                time.sleep(SLEEP_TIME)


class MmapGradSyncCallback(TrainerCallback):
    def __init__(self, model, grad_dir, gpu, gpus):
        self.model = model
        self.grad_dir = grad_dir
        self.gpu = gpu
        self.local_rank = gpus.index(gpu)
        self.gpus = gpus
        self.grad_sync = MmapGradientSync(model, gpu, gpus, grad_dir)

    def on_pre_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        step = state.global_step
        self.grad_sync.accumulate_local_grad(self.model, step)
        self.grad_sync.read_final_grad_into_model(self.model, step, average=True)

    def on_optimizer_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        step = state.global_step
        self.grad_sync.zero_mmaps(step)
        self.grad_sync.iteration_done(step)
