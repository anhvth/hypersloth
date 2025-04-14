import sys
import argparse
from loguru import logger
from speedy_utils.all import multi_process

import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Packing script")
parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file")
parser.add_argument("-o", "--output", required=True, help="Path to the output JSON file")
parser.add_argument("--seq_len", type=int, default=4096, help="Maximum sequence length")
parser.add_argument("--workers", type=int, default=128, help="Number of workers for multiprocessing")
args = parser.parse_args()

# Load the data
print(f"Loading data from {args.input}...")
with open(args.input, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")

MAX_SEQ_LEN = args.seq_len

model_name = "unsloth/gemma-3-27b-it"
# Load the Gemma tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
def item_to_length(item):
    """Calculates token length for an item and adds it as 'total_len'."""
    messages = None
    if isinstance(item, dict):
        if 'messages' in item:
            messages = item['messages']
        elif 'conversations' in item:
            messages = item['conversations']

    if not messages:
        item = item if isinstance(item, dict) else {} # Ensure item is a dict
        item['total_len'] = 0
        # Optionally log a warning:
        # print(f"Warning: No 'messages' or 'conversations' key found in item: {item}", file=sys.stderr)
        return item

    # Ensure item is a dict before modifying
    if not isinstance(item, dict):
         # If input wasn't a dict but we found messages somehow (unlikely given above checks)
         item = {'original_data': item} # Preserve original data if needed

    try:
        # Format text; assuming messages follow ChatML or similar structure expected by tokenizer
        # add_generation_prompt=False is crucial for accurate length calculation of existing text
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        # Calculate the number of tokens
        item['total_len'] = len(tokenizer.encode(formatted_text))
    except Exception as e:
        print(f"Warning: Error tokenizing item. Setting length to 0. Error: {e}. Item: {messages}", file=sys.stderr)
        item['total_len'] = 0 # Handle tokenization errors gracefully

    return item


def merge(item1, item2):
    """Merges two pre-processed items by combining messages and summing total_len."""
    # Basic validation relying on item_to_length having run successfully
    msg_key1 = 'messages' if 'messages' in item1 else 'conversations'
    msg_key2 = 'messages' if 'messages' in item2 else 'conversations'

    if msg_key1 not in item1 or msg_key2 not in item2 or \
       'total_len' not in item1 or 'total_len' not in item2:
        raise ValueError("Cannot merge items: keys ('messages'/'conversations' or 'total_len') missing.")

    if not isinstance(item1[msg_key1], list) or not isinstance(item2[msg_key2], list):
        raise TypeError("'messages'/'conversations' value must be a list.")

    # Combine messages and sum lengths
    combined_messages = item1[msg_key1] + item2[msg_key2]
    new_total_len = item1['total_len'] + item2['total_len']

    # Create the new item, standardizing on 'messages' key for output
    new_item = {
        'messages': combined_messages,
        'total_len': new_total_len
    }
    return new_item


def packing(items, max_seq_len=MAX_SEQ_LEN, shuffle=True):
    """Packs processed items into batches based on 'total_len'."""
    if not items:
        return []

    # Ensure items are processed (simple check on the first item)
    if 'total_len' not in items[0]:
         raise ValueError("Items must contain 'total_len'. Process with item_to_length first.")

    item_indices = list(range(len(items)))
    if shuffle:
        random.shuffle(item_indices)

    batches = []
    current_batch_items = []
    current_batch_len = 0
    ignore_too_long = 0

    for index in item_indices:
        item = items[index]
        # Use .get() for safety, default to 0 if somehow missing after first check
        item_len = item.get('total_len', 0)

        # Skip items that individually exceed max length
        if item_len > max_seq_len:
            ignore_too_long += 1
            continue

        # If the batch is empty or the new item fits
        if not current_batch_items or (current_batch_len + item_len <= max_seq_len):
            current_batch_items.append(item) # Add the entire item dict
            current_batch_len += item_len
        else:
            # Current item doesn't fit, finalize the current batch
            batches.append(current_batch_items)
            # Start a new batch with the current item
            current_batch_items = [item]
            current_batch_len = item_len

    # Add the last batch if it's not empty
    if current_batch_items:
        batches.append(current_batch_items)
    logger.info(f"Total batches: {len(batches)}")
    logger.info(f"Total items: {len(items)}")
    logger.warning(f"Total ignored: {ignore_too_long} items")

    # now merge same batch into one item with messages and total_len
    new_items = []
    for items in batches:
        # Merge the items in the batch
        merged_item = items[0]
        for item in items[1:]:
            merged_item = merge(merged_item, item)
        new_items.append(merged_item)
    # Total num of tokens
    total_tokens = sum(item['total_len'] for item in new_items)
    print(f"Total tokens in packed data: {total_tokens}")
    # save the distribution to /tmp/dist.png
    
    plt.figure(figsize=(10, 6))
    plt.hist([item['total_len'] for item in new_items], bins=30, alpha=0.7)
    plt.title('Distribution of Total Lengths in Packed Data')
    plt.xlabel('Total Length')
    plt.ylabel('Frequency')
    plt.savefig('/tmp/dist.png')
    plt.close()
    return new_items



items_with_lens = multi_process(item_to_length, data, workers=args.workers)
items_packed = packing(items_with_lens, max_seq_len=MAX_SEQ_LEN, shuffle=True)

# Save the packed data to the output file
print(f"Saving packed data to {args.output}...")
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(items_packed, f, ensure_ascii=False, indent=4)