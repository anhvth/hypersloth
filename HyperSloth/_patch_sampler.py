import os
import random
from typing import Iterator

from datasets import Dataset
from fastcore.all import patch
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainerCallback
from .logging_config import get_safe_logger

# Use safe logger that handles gpu_id properly
logger = get_safe_logger()


def _compute_reordered_and_shuffled_ids(
    dataset: Dataset,
    epoch,
    seed=42,
) -> list[int]:
    from fastcore.all import chunked

    lens = [len(x["input_ids"]) for x in dataset]
    sorted_ids = sorted(range(len(lens)), key=lambda k: lens[k])

    global_bz = int(os.environ["HYPERSLOTH_GLOBAL_BATCH_SIZE"])
    chunked_ids = list(chunked(sorted_ids, global_bz))
    random.Random(seed + epoch).shuffle(chunked_ids)

    # Log first 10 chunks for debugging
    for i in range(min(10, len(chunked_ids))):
        chunk = chunked_ids[i]
        chunk_lens = [lens[idx] for idx in chunk]
        logger.info(
            f"[{i}] Chunk {i} length: {len(chunk)} "
            f"with mean length {sum(chunk_lens) / len(chunk_lens):.1f} "
            f"then min length {min(chunk_lens)} "
            f"and max length {max(chunk_lens)}"
        )

    return [idx for chunk in chunked_ids for idx in chunk]


def reorder_and_shuffle_data(
    dataset: Dataset,
    epoch,
    seed=42,
) -> Dataset:
    ids = _compute_reordered_and_shuffled_ids(dataset, epoch, seed)
    dataset = dataset.select(ids)
    return dataset


def print_sequence_lengths(dataset: Dataset):
    lens = [len(x["input_ids"]) for x in dataset]
    logger.info(f"First 10 sequence lengths: {lens[:10]}")
    logger.info(f"Last 10 sequence lengths: {lens[-10:]}")
    logger.info(f"Max sequence length: {max(lens)}")
    logger.info(f"Min sequence length: {min(lens)}")
    logger.info(f"Mean sequence length: {sum(lens) / len(lens)}")


def get_callback_shuffle_data(trainer) -> TrainerCallback:
    "return a callback to shuffle data on_epoch_begin"

    class ShuffleData(TrainerCallback):
        def __init__(self, trainer):
            self.trainer: Trainer = trainer

        def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
            local_rank = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
            logger.info("[on_epoch_begin] Shuffling data, this may take a while...")

            # self.trainer.train_dataset = reorder_and_shuffle_data(
            #     self.trainer.train_dataset,
            #     epoch=state.epoch,
            #     seed=args.seed,
            # )
            logger.info("[on_epoch_begin] Data shuffled")

            # print_sequence_lengths(self.trainer.train_dataset)

            if local_rank == 0:
                logger.info("[on_epoch_begin] Debugging dataloader")
                try:
                    from ._debug_dataloader import _debug_dataloader

                    tok = kwargs["processing_class"]
                    _debug_dataloader(
                        self.trainer.get_train_dataloader(), tokenizer=tok
                    )
                except:
                    logger.exception("Failed to debug dataloader this is not a problem")
                    pass
            logger.info("[on_epoch_begin] Finished debugging dataloader")

    return ShuffleData(trainer)


from torch.utils.data.sampler import SequentialSampler


class CustomSampler(SequentialSampler):
    def __init__(self, data_source) -> None:
        self.data_source = data_source
        self.ids = _compute_reordered_and_shuffled_ids(
            data_source,
            epoch=0,
            seed=42,
        )

    def __iter__(self) -> Iterator[int]:
        return iter(self.ids)


from speedy_utils import Clock


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
