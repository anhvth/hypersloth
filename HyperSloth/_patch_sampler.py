import os
import random
import statistics
from typing import Iterator

from datasets import Dataset
from fastcore.all import patch
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainerCallback
from loguru import logger


def _compute_reordered_and_shuffled_ids(
    dataset: Dataset,
    epoch,
    seed=42,
    print_debug_table: bool = True,
) -> list[int]:
    from fastcore.all import chunked

    lens = [len(x["input_ids"]) for x in dataset]
    sorted_ids = sorted(range(len(lens)), key=lambda k: lens[k])

    global_bz = int(os.environ["HYPERSLOTH_GPU_FORWARD_BATCH_SIZE"])
    chunked_ids = list(chunked(sorted_ids, global_bz))
    random.Random(seed + epoch).shuffle(chunked_ids)

    # Log first 10 chunks as a table for debugging
    if chunked_ids and print_debug_table:

        log_stats(lens, chunked_ids)

    return [idx for chunk in chunked_ids for idx in chunk]


def log_stats(lens, chunked_ids):
    table_data = []
    random_batch_ids: list[int] = random.sample(
        range(len(chunked_ids)), min(10, len(chunked_ids))
    )
    for i in random_batch_ids:
        chunk = chunked_ids[i]
        chunk_lens = [lens[idx] for idx in chunk]
        mean_len = sum(chunk_lens) / len(chunk_lens)
        min_len = min(chunk_lens)
        max_len = max(chunk_lens)
        chunk_lens_norm = [
            (l - mean_len) / mean_len for l in chunk_lens if mean_len > 0
        ]
        std_len = statistics.stdev(chunk_lens_norm) if len(chunk_lens) > 1 else 0.0
        table_data.append(
            [i, len(chunk), f"{mean_len:.1f}", min_len, max_len, f"{std_len:.1f}"]
        )

    headers = [
        "Batch Idx",
        "BatchSize",
        "MeanSqlLen",
        "MinSqlLen",
        "MaxSqlLen",
        "StdSqlLen",
    ]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    logger.info(f"Random batches debugging table:\n{table}")


def reorder_and_shuffle_data(
    dataset: Dataset,
    epoch,
    seed=42,
) -> Dataset:
    ids = _compute_reordered_and_shuffled_ids(dataset, epoch, seed)
    dataset = dataset.select(ids)
    return dataset


# def print_sequence_lengths(dataset: Dataset):
#     lens = [len(x["input_ids"]) for x in dataset]
#     logger.info(f"First 10 sequence lengths: {lens[:10]}")
#     logger.info(f"Last 10 sequence lengths: {lens[-10:]}")
#     logger.info(f"Max sequence length: {max(lens)}")
#     logger.info(f"Min sequence length: {min(lens)}")
#     logger.info(f"Mean sequence length: {sum(lens) / len(lens)}")


def get_callback_shuffle_data(trainer) -> TrainerCallback:
    "return a callback to shuffle data on_epoch_begin"

    class ShuffleData(TrainerCallback):
        def __init__(self, trainer):
            self.trainer: Trainer = trainer

        def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
            local_rank = int(os.environ["HYPERSLOTH_LOCAL_RANK"])

            # Group shuffling operations into single log message
            logger.info(
                f"[Epoch {state.epoch}] Starting data shuffle and sampler update..."
            )

            self.trainer.train_dataset = reorder_and_shuffle_data(
                self.trainer.train_dataset,
                epoch=state.epoch,
                seed=args.seed,
            )

            # Update sampler epoch if it exists
            train_sampler = getattr(train_dataloader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(state.epoch)

            logger.info(
                f"[Epoch {state.epoch}] Data shuffle and sampler update complete"
            )

            # Only debug on rank 0 and make it less verbose
            if local_rank == 0:
                try:
                    from ._debug_dataloader import _debug_dataloader

                    tok = kwargs["processing_class"]
                    _debug_dataloader(
                        self.trainer.get_train_dataloader(), tokenizer=tok
                    )
                    logger.info(f"[Epoch {state.epoch}] Dataloader debug complete")
                except Exception:
                    logger.warning(
                        f"[Epoch {state.epoch}] Dataloader debug failed (non-critical)"
                    )

    return ShuffleData(trainer)


from torch.utils.data.sampler import SequentialSampler


class CustomSampler(SequentialSampler):
    def __init__(self, data_source) -> None:
        self.data_source = data_source
        self.epoch = 0
        self.seed = 42
        self.ids = _compute_reordered_and_shuffled_ids(
            data_source,
            epoch=self.epoch,
            seed=self.seed,
        )

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch and recompute ids."""
        self.epoch = epoch
        self.ids = _compute_reordered_and_shuffled_ids(
            self.data_source,
            epoch=self.epoch,
            seed=self.seed,
        )

    def __iter__(self) -> Iterator[int]:
        return iter(self.ids)


from speedy_utils import Clock
from tabulate import tabulate


def patch_sampler(trainer: Trainer):
    clock = Clock(start_now=True)

    @patch
    def _get_train_sampler(self: Trainer) -> CustomSampler:
        """Get a custom sampler for the training dataset."""
        logger.info(f"Total samples in dataset: {len(self.train_dataset)}")
        return CustomSampler(self.train_dataset)

    trainer.add_callback(get_callback_shuffle_data(trainer))
    clock.log_elapsed_time()
    return trainer
