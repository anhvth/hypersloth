import random
from typing import Any

from datasets import Dataset

import warnings
import os
from datasets import load_dataset
from typing import Any

from datasets import load_dataset
from speedy_utils import load_by_ext, log, multi_thread
from speedy_utils import jloads, jdumps

warnings.filterwarnings("ignore", category=UserWarning)
R = random.Random(42)  # Set a random seed for reproducibility


def shuffle_one_messages(messages):
    ratio = 0.5
    if random.random() < ratio:
        return messages
    # find the where the turn role is user and content is json string then use jloads to load it->dict, then shuffle the keys (augmentation)
    for message in messages:
        if message["role"] == "user":
            try:
                message["content"] = jloads(message["content"])
                # shuffle the keys
                keys = list(message["content"].keys())
                if len(keys) > 1:
                    shuffled_keys = keys[:]
                    old_order = shuffled_keys[:]
                    random.shuffle(shuffled_keys)
                    new_order = shuffled_keys[:]
                    log(
                        f"Shuffled keys: {old_order} -> {new_order}",
                        level="info",
                        once=True,
                    )
                    message["content"] = jdumps(
                        {k: message["content"][k] for k in shuffled_keys}
                    )
            except Exception as e:
                log(f"Error while shuffling dict keys: {e}", level="warning", once=True)
    return messages


def get_chat_dataset(
    dataset_name_or_path: str,
    split: str = None,
    num_samples: int = None,
    test_ratio: float = 0,
    tokenizer: Any = None,
    message_key: str = None,
    chat_template=None,
    dataset_already_formated=False,  # when there already a "text" key in the dataset
    shuffle_user_dict_keys=False,
    **kwargs,
) -> tuple[Dataset, Dataset | None]:
    """
    Load and preprocess a chat dataset from file or HuggingFace Hub.

    Returns train dataset and optional test dataset (if test_ratio > 0).
    """
    if isinstance(dataset_name_or_path, list):
        # load one by one then merge
        train_datasets = []
        test_datasets = []
        for dataset_path in dataset_name_or_path:
            train_dataset, test_dataset = get_chat_dataset(
                dataset_path,
                split=split,
                num_samples=num_samples,
                test_ratio=test_ratio,
                tokenizer=tokenizer,
                message_key=message_key,
                chat_template=chat_template,
                dataset_already_formated=dataset_already_formated,
                **kwargs,
            )
            train_datasets.append(train_dataset.to_list())
            if test_dataset:
                test_datasets.append(test_dataset.to_list())
        # # Merge datasets
        ds_train = Dataset.from_list(
            [item for sublist in train_datasets for item in sublist]
        )
        if test_datasets:
            ds_test = Dataset.from_list(
                [item for sublist in test_datasets for item in sublist]
            )
        else:
            ds_test = None
        return ds_train, ds_test
        # return ds_train, ds_test
    # Load dataset based on input type
    if os.path.exists(dataset_name_or_path):
        if dataset_name_or_path.endswith(".json"):
            dataset = Dataset.from_list(load_by_ext(dataset_name_or_path))
        elif dataset_name_or_path.endswith(".csv"):
            dataset = Dataset.from_csv(dataset_name_or_path)
        # is folder
        elif os.path.isdir(dataset_name_or_path):
            dataset = Dataset.load_from_disk(dataset_name_or_path)
        else:
            raise ValueError(
                f"Unsupported dataset file format: {dataset_name_or_path}. "
                "Supported formats: .json, .csv, or directory. "
                "Example notebook preparing dataset: https://github.com/anhvth/hypersloth/blob/main/nbs/prepare_dataset_example.ipynb"
            )
    else:
        try:
            from unsloth_zoo.dataset_utils import standardize_data_formats

            dataset = load_dataset(dataset_name_or_path, split=split)
            # Check if dataset is empty
            dataset[0]
            dataset = standardize_data_formats(dataset)

        except IndexError:
            raise ValueError(
                f"Dataset is empty. Check dataset name and split: {dataset_name_or_path}, split={split}"
            )
        except Exception as e:
            if split is None:
                raise ValueError(
                    f"Failed to load dataset '{dataset_name_or_path}'. You might need to specify a split."
                ) from e
            else:
                raise ValueError(
                    f"Failed to load dataset '{dataset_name_or_path}' with split '{split}': {str(e)}"
                ) from e
    if dataset_already_formated:
        assert "text" in dataset[0], "Dataset already formated, but no 'text' key found"
        return dataset, None
    example = dataset[0]
    # Show dataset structure for debugging
    dataset_keys = list(example.keys())
    if num_samples:
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(num_samples))
    if shuffle_user_dict_keys or 1:
        # Shuffle the keys of the user dict in the messages
        log("Shuffling the keys of the user dict in the messages", level="info")
        dataset = dataset.map(
            lambda x: {"messages": shuffle_user_dict_keys(x["messages"])},
            batched=True,
            remove_columns=["messages"],
        )
    if tokenizer:

        def apply_chat_template(examples):
            # Use custom message key if provided
            if message_key and message_key in examples:
                messages_key = message_key
            # Try to detect the correct key for messages
            elif "messages" in examples:
                messages_key = "messages"
            elif "conversations" in examples:
                messages_key = "conversations"
            else:
                available_keys = list(examples.keys())
                raise ValueError(
                    "Dataset does not contain recognized message keys ('messages' or 'conversations'). "
                    f"Available keys: {available_keys}. "
                    "Use the message_key parameter to specify the correct key. "
                    "Expected chat format example: (mlabonne/FineTome-100k) https://huggingface.co/datasets/mlabonne/FineTome-100k"
                )
            if chat_template:
                from transformers import AutoTokenizer

                tokenizer.chat_template = AutoTokenizer.from_pretrained(
                    chat_template
                ).chat_template

            texts = tokenizer.apply_chat_template(
                examples[messages_key], tokenize=False
            )
            return {"text": texts}

        try:
            dataset = dataset.map(apply_chat_template, batched=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise ValueError(
                f"Failed to apply chat template: {str(e)}\n"
                f"Dataset structure: Keys in first example = {dataset_keys}\n"
                f"Make sure the dataset has conversation data in a recognized format."
            ) from e

    # Create train/test split if needed
    if test_ratio > 0:
        ds = dataset.train_test_split(test_size=test_ratio, shuffle=True, seed=42)
        ds_train, ds_test = ds["train"], ds["test"]
        log(
            f"Splitting dataset into train and test sets, test_ratio={test_ratio}, seed=42, Num test samples={len(ds_test)}"
        )
    else:
        ds_train, ds_test = dataset, None

    return ds_train, ds_test
