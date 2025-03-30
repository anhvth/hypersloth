import os
import random
from datasets import Dataset
from fastcore.all import patch
from loguru import logger
from torch.utils.data import SequentialSampler
from transformers import Trainer


def reorder_and_shuffle_data(
    dataset: Dataset,
    epoch,
    seed=42,
):

    lens = [len(x["input_ids"]) for x in dataset]
    sorted_ids = sorted(range(len(lens)), key=lambda k: lens[k])
    dataset = dataset.select(sorted_ids)

    from fastcore.all import chunked

    num_gpus = int(os.environ["HYPERSLOTH_NUM_GPUS"])
    num_gpus = 5
    
    # group size
    chunked_lens = list(
        chunked(
            range(len(lens)),
           num_gpus,
        )
    )
    random.Random(seed + epoch).shuffle(
        chunked_lens
    )  # the 8 continous value are similar
    ids = [idx for chunk in chunked_lens for idx in chunk]
    dataset = dataset.select(ids)
    return dataset


def print_sequence_lengths(dataset: Dataset):
    lens = [len(x["input_ids"]) for x in dataset]
    logger.info(f"First 10 sequence lengths: {lens[:10]}")
    logger.info(f"Last 10 sequence lengths: {lens[-10:]}")
    logger.info(f"Max sequence length: {max(lens)}")
    logger.info(f"Min sequence length: {min(lens)}")
    logger.info(f"Mean sequence length: {sum(lens) / len(lens)}")


# callback to shuffle data on_epoch_begin
from transformers import TrainerCallback


def get_callback_shuffle_data(trainer) -> TrainerCallback:
    "return a callback to shuffle data on_epoch_begin"

    class ShuffleData(TrainerCallback):
        def __init__(self, trainer):
            self.trainer: Trainer = trainer

        def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
            # if state.epoch == 0:
            #     return
            local_rank = int(os.environ["HYPERSLOTH_LOCAL_RANK"])
            # Debug info for the main GPU

            logger.info("[on_epoch_begin] Shuffling data, this may take a while...")
            self.trainer.train_dataset = reorder_and_shuffle_data(
                self.trainer.train_dataset,
                epoch=state.epoch,
                seed=args.seed,
            )
            logger.info("[on_epoch_begin] Data shuffled")
            print_sequence_lengths(self.trainer.train_dataset)

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


def patch_sampler(trainer: Trainer):

    @patch
    def _get_train_sampler(self: Trainer) -> SequentialSampler:
        """Get a sequential sampler for the training dataset."""
        return SequentialSampler(self.train_dataset)

    trainer.train_dataset = reorder_and_shuffle_data(
        trainer.train_dataset,
        epoch=0,
        seed=trainer.args.seed,
    )
    trainer.add_callback(get_callback_shuffle_data(trainer))

    return trainer
