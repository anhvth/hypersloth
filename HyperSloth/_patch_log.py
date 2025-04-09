import os
import time
from typing import Dict, Optional

import numpy as np
from fastcore.all import patch
from loguru import logger
from transformers.trainer import (
    Trainer,
    speed_metrics,
)
SLEEP_TIME = 0.05


def _patch_log(T: type):
    support_keys = [
        "loss",
        "grad_norm",
        "num_input_tokens_seen",
        "trained_token_ratio",
    ]
    LOG_MMAP = {}
    HYPERSLOTH_LOCAL_RANK = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
    n = int(os.environ["HYPERSLOTH_NUM_GPUS"])
    logger.info("Num GPUs: {}".format(n), once=True)
            
    HYPERSLOTH_RUN_DIR = os.environ["HYPERSLOTH_RUN_DIR"]
    for key in support_keys:
        LOG_MMAP[key] = np.memmap(
            f"{HYPERSLOTH_RUN_DIR}/log_{key}.mmap",
            dtype="float32",
            mode="w+",
            shape=(n,),
        )

    @patch
    def log(
        self:T, logs: Dict[str, float], start_time: Optional[float] = None # type: ignore
    ) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics(
                    "train", start_time, num_tokens=self.state.num_input_tokens_seen
                )
        # ===HYPER SLOTH >>>>
        if hasattr(self.state, "trained_token_ratio"):
            logs["trained_token_ratio"] = self.state.trained_token_ratio

        output = {**logs, **{"step": self.state.global_step}}

        is_main = HYPERSLOTH_LOCAL_RANK == 0
        self.state.log_history.append(output)
        if "loss" in logs:
            for k in support_keys:
                LOG_MMAP[k][HYPERSLOTH_LOCAL_RANK] = logs[k]

            is_done = lambda: (LOG_MMAP["loss"][:] != 0).all()
            if is_main:
                while not is_done():
                    time.sleep(SLEEP_TIME)
                # perform summing or avg then do log on main process
                logs["loss"] = float(LOG_MMAP["loss"].sum())
                logs["grad_norm"] = float(LOG_MMAP["grad_norm"].sum())
                logs["num_input_tokens_seen"] = int(
                    LOG_MMAP["num_input_tokens_seen"].sum()
                )
                logs["trained_token_ratio"] = float(
                    LOG_MMAP["trained_token_ratio"].mean()
                )

                self.control = self.callback_handler.on_log(
                    self.args, self.state, self.control, logs
                )
                LOG_MMAP["loss"][:] = 0
            else:
                # the condition is when the main process reset losses to 0
                while not (LOG_MMAP["loss"] == 0).all():
                    logger.debug(
                        f"[{HYPERSLOTH_LOCAL_RANK=}] Waiting for the main process to finish logging"
                    )
                    time.sleep(SLEEP_TIME)

        elif is_main:
            self.control = self.callback_handler.on_log(
                self.args, self.state, self.control, logs
            )
