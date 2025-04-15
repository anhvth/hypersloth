import os
import time
from typing import Dict, Optional
import numpy as np
from filelock import FileLock # Import FileLock
from fastcore.all import patch
from loguru import logger
from transformers.trainer import (
    Trainer,
    speed_metrics,
)
# Assuming the Flag class from the 'safe lock' example is available
# If not, copy the Flag class definition here.
# <safe lock> code (Flag class definition) should be placed here or imported
# --- Start of Flag class definition (copied from 'safe lock') ---
import os
import time
import numpy as np
import torch
from filelock import FileLock
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from loguru import logger

TIME_OUT = 300
SLEEP_TIME = 0.01 # Use a small sleep time for waiting loops
WAIT_WARNING_THRESHOLD = 2  # Log a warning if waiting longer than this


class Flag:
    def __init__(self, world_size: int, file_path: str, is_master: bool = False):
        self.world_size = world_size
        self.file_path = file_path
        self.is_master = is_master
        self.lock_path = self.file_path + ".lock"
        self.lock = FileLock(self.lock_path)

        # Master process creates the directory and the file
        if self.is_master:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Ensure file exists and has the correct size before memmap
            with self.lock: # Lock during file creation/truncation
                with open(self.file_path, "wb") as f:
                    f.truncate(world_size * 4) # 4 bytes per float32
        else:
            # Worker processes wait for the file to be created by the master
            t0 = time.time()
            while not os.path.exists(self.file_path):
                time.sleep(SLEEP_TIME)
                if time.time() - t0 > TIME_OUT:
                     raise TimeoutError(f"Worker timed out waiting for flag file {self.file_path}")


        # All processes create the memmap
        # Need to ensure file is ready and has correct size before mapping
        file_ready = False
        t0 = time.time()
        while not file_ready:
             try:
                 # Check file size to ensure master has finished truncating
                 if os.path.getsize(self.file_path) == self.world_size * 4:
                     self.mem = np.memmap(
                         self.file_path,
                         dtype="float32",
                         mode="r+",
                         shape=(self.world_size,),
                     )
                     file_ready = True
                 else:
                      time.sleep(SLEEP_TIME)
             except FileNotFoundError:
                 time.sleep(SLEEP_TIME) # File might not be visible yet
             except Exception as e:
                 logger.error(f"Error creating memmap for {self.file_path}: {e}")
                 time.sleep(SLEEP_TIME*10) # Wait longer if error

             if time.time() - t0 > TIME_OUT and not file_ready:
                 raise TimeoutError(f"Timeout waiting for correct size or mapping flag file {self.file_path}. "
                                    f"Expected size: {self.world_size * 4}, Found: {os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 'Not Found'}")


        if self.is_master:
            self.reset()

    def update(self, rank: int):
        with self.lock:
            self.mem[rank] = 1.0
            self.mem.flush()

    def wait_for_all(self, step: int = -1, timeout: float = TIME_OUT): # Add default step
        t0 = time.time()
        has_logged = False

        while True:
            all_set = False
            with self.lock:
                # Read the current state safely
                flags_copy = self.mem.copy()
                if np.all(flags_copy == 1.0):
                   all_set = True

            if all_set:
                break # Exit loop if all flags are set

            # Check for timeout and log warnings outside the lock
            elapsed = time.time() - t0
            if elapsed > WAIT_WARNING_THRESHOLD and not has_logged:
                # Log outside lock to avoid holding it while logging
                logger.warning(f"[Flag={self.file_path}] waiting {elapsed:.1f}s at step={step}, flags={flags_copy.tolist()}")
                has_logged = True

            if elapsed > timeout:
                # Log outside lock
                raise RuntimeError(f"[Flag={self.file_path}] Timeout after {elapsed:.1f}s waiting at step={step}, flags={flags_copy.tolist()}")

            time.sleep(SLEEP_TIME) # Sleep outside the lock

        # Optional: Debug log when successful (outside lock)
        # logger.debug(f"[Flag={self.file_path}] all ranks ready at step={step}")


    def wait_for_reset(self, rank: int, step: int = -1, timeout: float = TIME_OUT):
        """Wait until the flag for the given rank is reset (becomes 0.0)."""
        t0 = time.time()
        has_logged = False
        while True:
            is_reset = False
            current_val = -1.0 # Default invalid value
            with self.lock:
                current_val = self.mem[rank]
                if current_val == 0.0:
                    is_reset = True

            if is_reset:
                break # Exit loop if flag is reset

            elapsed = time.time() - t0
            if elapsed > WAIT_WARNING_THRESHOLD and not has_logged:
                 logger.warning(f"[Flag={self.file_path}] rank={rank} waiting for reset {elapsed:.1f}s at step={step}, current_val={current_val}")
                 has_logged = True

            if elapsed > timeout:
                raise RuntimeError(f"[Flag={self.file_path}] rank={rank} Timeout after {elapsed:.1f}s waiting for reset at step={step}, current_val={current_val}")

            time.sleep(SLEEP_TIME)


    def reset(self):
        if not self.is_master:
            # This check might be overly strict if workers might need to reset in some scenarios,
            # but for this specific log patching, only master should reset.
            raise RuntimeError("Only master can reset a flag array.")
        with self.lock:
            self.mem[:] = 0.0
            self.mem.flush()
        # logger.debug(f"[Flag={self.file_path}] Master reset flags.")

# --- End of Flag class definition ---


# Define constants consistent with Flag class
# SLEEP_TIME = 0.01 # Already defined in Flag class import


def _patch_log(T: type):
    support_keys = [
        "loss",
        "grad_norm",
        "num_input_tokens_seen",
        "trained_token_ratio",
    ]
    LOG_MMAP: Dict[str, np.memmap] = {}
    LOG_LOCKS: Dict[str, FileLock] = {} # Dictionary for locks

    # --- Initialization (runs once when the patch is applied) ---
    try:
        HYPERSLOTH_LOCAL_RANK = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
        n = int(os.environ["HYPERSLOTH_NUM_GPUS"])
        HYPERSLOTH_RUN_DIR = os.environ["HYPERSLOTH_RUN_DIR"]
        is_main = HYPERSLOTH_LOCAL_RANK == 0

        logger.info(f"[{HYPERSLOTH_LOCAL_RANK=}] Patching log function. Run Dir: {HYPERSLOTH_RUN_DIR}, Num GPUs: {n}")

        # Create run directory if it doesn't exist (master should do this)
        if is_main:
             os.makedirs(HYPERSLOTH_RUN_DIR, exist_ok=True)
        else:
             # Workers wait briefly for the directory to be created
             t0 = time.time()
             while not os.path.exists(HYPERSLOTH_RUN_DIR):
                 time.sleep(SLEEP_TIME)
                 if time.time() - t0 > 60: # 1 minute timeout for dir creation
                      raise TimeoutError(f"Worker {HYPERSLOTH_LOCAL_RANK} timed out waiting for run directory {HYPERSLOTH_RUN_DIR}")


        # Initialize mmaps and locks
        for key in support_keys:
            mmap_path = f"{HYPERSLOTH_RUN_DIR}/log_{key}.mmap"
            lock_path = mmap_path + ".lock"
            LOG_LOCKS[key] = FileLock(lock_path)

            # Master creates/truncates the file
            if is_main:
                 with LOG_LOCKS[key]:
                     # logger.debug(f"Master creating/truncating {mmap_path} for {n} GPUs.")
                     with open(mmap_path, "wb") as f:
                         f.truncate(n * 4) # float32 = 4 bytes
            else:
                 # Workers wait for file existence
                 t0 = time.time()
                 while not os.path.exists(mmap_path):
                     time.sleep(SLEEP_TIME)
                     if time.time() - t0 > TIME_OUT:
                         raise TimeoutError(f"Worker {HYPERSLOTH_LOCAL_RANK} timed out waiting for mmap file {mmap_path}")

            # All processes create the memmap (ensure file size is correct first)
            file_ready = False
            t0 = time.time()
            while not file_ready:
                try:
                    expected_size = n * 4
                    if os.path.exists(mmap_path) and os.path.getsize(mmap_path) == expected_size:
                        LOG_MMAP[key] = np.memmap(
                            mmap_path,
                            dtype="float32",
                            mode="r+",
                            shape=(n,),
                        )
                        file_ready = True
                        # logger.debug(f"Rank {HYPERSLOTH_LOCAL_RANK} successfully mapped {mmap_path}")
                    else:
                         # # logger.debug(f"Rank {HYPERSLOTH_LOCAL_RANK} waiting for {mmap_path}. Size: {os.path.getsize(mmap_path) if os.path.exists(mmap_path) else 'N/A'}, Expected: {expected_size}")
                         time.sleep(SLEEP_TIME)
                except FileNotFoundError:
                    time.sleep(SLEEP_TIME)
                except Exception as e:
                     logger.error(f"Rank {HYPERSLOTH_LOCAL_RANK} error mapping {mmap_path}: {e}. Retrying...")
                     time.sleep(SLEEP_TIME*10) # Wait longer on error

                if time.time() - t0 > TIME_OUT and not file_ready:
                     raise TimeoutError(f"Rank {HYPERSLOTH_LOCAL_RANK} timeout waiting for correct size or mapping {mmap_path}. "
                                        f"Expected size: {expected_size}, Found: {os.path.getsize(mmap_path) if os.path.exists(mmap_path) else 'Not Found'}")


            # Master initializes the mmap to zero
            if is_main:
                 with LOG_LOCKS[key]:
                     LOG_MMAP[key][:] = 0.0
                     LOG_MMAP[key].flush()

        # Initialize the synchronization Flag
        log_sync_flag_path = f"{HYPERSLOTH_RUN_DIR}/log_sync_flag.dat"
        log_sync_flag = Flag(world_size=n, file_path=log_sync_flag_path, is_master=is_main)
        logger.info(f"[{HYPERSLOTH_LOCAL_RANK=}] Log patch initialization complete.")

    except Exception as e:
         logger.error(f"[{HYPERSLOTH_LOCAL_RANK=}] CRITICAL ERROR during log patch initialization: {e}")
         # Depending on the desired behavior, you might want to raise the exception
         # to halt the process or log and attempt to continue (though logging might fail).
         raise e # Re-raise to make the failure explicit

    # --- Patched log method definition ---
    @patch
    def log(
        self:T, logs: Dict[str, float], start_time: Optional[float] = None # type: ignore
    ) -> None:
        """
        Log `logs` on the various objects watching training. (Patched for mmap sync)

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        # Standard HuggingFace logging setup
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            # Note: speed_metrics might involve communication if not careful,
            # but it's usually called only on the main process by default trainer.
            # If called here by all processes, ensure it's safe for multiprocessing.
            # Original code calls it without checks, assuming it's safe or handled internally.
            if start_time is not None and is_main: # Safest to call only on main
                 speed_metrics(
                     "train", start_time, num_tokens=self.state.num_input_tokens_seen
                 )
        # ===HYPER SLOTH specific field >>>>
        if hasattr(self.state, "trained_token_ratio"):
            logs["trained_token_ratio"] = self.state.trained_token_ratio

        output = {**logs, **{"step": self.state.global_step}}
        current_step = self.state.global_step # For logging/debugging clarity

        # Append to local log history regardless of synchronization
        self.state.log_history.append(output)

        # --- Synchronization logic for training steps (identified by 'loss' key) ---
        if "loss" in logs:
            # 1. Write local log data to mmap (with locks)
            try:
                for key in support_keys:
                    if key in logs: # Only write keys present in this log call
                        with LOG_LOCKS[key]:
                            LOG_MMAP[key][HYPERSLOTH_LOCAL_RANK] = logs[key]
                            LOG_MMAP[key].flush()
                    # else: # Handle missing keys if necessary (e.g., write 0 or NaN)
                    #    with LOG_LOCKS[key]:
                    #       LOG_MMAP[key][HYPERSLOTH_LOCAL_RANK] = 0.0 # Or np.nan
                    #       LOG_MMAP[key].flush()
                # logger.debug(f"Rank {HYPERSLOTH_LOCAL_RANK} wrote logs to mmap at step {current_step}")

                # 2. Signal that this rank has written its data
                log_sync_flag.update(HYPERSLOTH_LOCAL_RANK)
                # logger.debug(f"Rank {HYPERSLOTH_LOCAL_RANK} updated log sync flag at step {current_step}")

            except Exception as e:
                logger.error(f"Rank {HYPERSLOTH_LOCAL_RANK} failed during mmap write or flag update at step {current_step}: {e}")
                # Handle error appropriately - maybe try to continue or raise

            # 3. Synchronize, Aggregate (main), and Wait (workers)
            if is_main:
                try:
                    # Wait for all ranks to write and signal
                    # logger.debug(f"Master waiting for all ranks on log sync flag at step {current_step}")
                    log_sync_flag.wait_for_all(step=current_step)
                    # logger.debug(f"Master confirmed all ranks ready at step {current_step}")

                    # Aggregate results from mmaps (with locks)
                    aggregated_logs = logs.copy() # Start with local logs (contains epoch, etc.)
                    for key in support_keys:
                         with LOG_LOCKS[key]:
                             all_vals = LOG_MMAP[key].copy() # Read all values under lock
                         # Decide aggregation strategy (sum for counts, mean for ratios/losses)
                         if key in ["num_input_tokens_seen"]:
                             aggregated_logs[key] = int(all_vals.sum())
                         elif key in ["trained_token_ratio"]:
                              # Use nanmean if zeros should be ignored, or simple mean if zeros are valid
                              valid_vals = all_vals[all_vals != 0.0] # Example: Ignore ranks that didn't log this
                              aggregated_logs[key] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
                              # aggregated_logs[key] = float(np.mean(all_vals)) # Simpler mean
                         else: # Default: sum (e.g., loss, grad_norm - assuming additive makes sense)
                             aggregated_logs[key] = float(all_vals.sum())

                    # logger.debug(f"Master aggregated logs at step {current_step}: {aggregated_logs}")


                    # Call the actual logging handlers (TensorBoard, etc.) with aggregated logs
                    self.control = self.callback_handler.on_log(
                        self.args, self.state, self.control, aggregated_logs # Use aggregated logs
                    )

                    # Reset mmaps to zero (with locks)
                    for key in support_keys:
                        with LOG_LOCKS[key]:
                            LOG_MMAP[key][:] = 0.0
                            LOG_MMAP[key].flush()

                    # Reset the flag to signal workers they can proceed
                    log_sync_flag.reset()
                    # logger.debug(f"Master finished logging and reset flag at step {current_step}")

                except Exception as e:
                    logger.error(f"Master failed during log aggregation/reset at step {current_step}: {e}")
                     # Attempt to reset flag anyway to potentially unblock workers, but log the error
                    try:
                        log_sync_flag.reset()
                        logger.warning(f"Master attempted flag reset after error at step {current_step}")
                    except Exception as reset_e:
                         logger.error(f"Master failed even to reset flag after error at step {current_step}: {reset_e}")
                    # Consider raising the original error e

            else: # Worker process
                try:
                    # Wait for the master to finish processing and reset the flag
                    # logger.debug(f"Worker {HYPERSLOTH_LOCAL_RANK} waiting for flag reset at step {current_step}")
                    log_sync_flag.wait_for_reset(rank=HYPERSLOTH_LOCAL_RANK, step=current_step)
                    # logger.debug(f"Worker {HYPERSLOTH_LOCAL_RANK} detected flag reset at step {current_step}")
                except Exception as e:
                    logger.error(f"Worker {HYPERSLOTH_LOCAL_RANK} failed waiting for flag reset at step {current_step}: {e}")
                    # Handle error - worker might be stuck or proceed with inconsistent state

        # --- Handle non-training logs (e.g., evaluation) ---
        # These typically don't need aggregation across ranks in the same way.
        # The original code calls on_log only on the main process for these.
        elif is_main:
             # logger.debug(f"Master logging non-training step {current_step}: {logs}")
             self.control = self.callback_handler.on_log(
                 self.args, self.state, self.control, logs
             )

    # Return the modified class type with the patched method
    return T

# --- Example Usage (assuming Trainer class exists) ---
# PatchedTrainer = _patch_log(Trainer)
# Now, instantiate PatchedTrainer instead of Trainer
# trainer = PatchedTrainer(...)
# trainer.train()