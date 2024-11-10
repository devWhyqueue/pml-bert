import os

import datasets as ds

from configuration.config import get_logger
from data.training.mlm import mask_batch
from data.training.preprocess import concatenate, break_down_into_spans_and_tokenize

log = get_logger(__name__)


def combine_datasets(datasets: list[ds.Dataset], destination: str, keep: list[str], shards: int, shuffle: bool):
    """
    Combine multiple datasets into one, optionally shuffle, and save to disk.

    Args:
        datasets (list[ds.Dataset]): List of datasets to combine.
        destination (str): Path to save the combined dataset.
        keep (list[str]): List of columns to keep in the combined dataset. Defaults to None (keeps all).
        shards (int): Number of shards to use when saving the dataset.
        shuffle (bool): Whether to shuffle the combined dataset.

    Returns:
        None
    """
    log.info(f"Unifying {len(datasets)} datasets...")
    combined = concatenate(datasets, keep)
    if shuffle:
        log.info("Shuffling unified dataset...")
        combined = combined.shuffle(seed=42)

    log.info(f"Saving unified dataset to {destination}...")
    combined.save_to_disk(destination, num_proc=shards)
    log.info(f"Saved unified dataset to {destination}.")


def spanify_and_tokenize(dataset_dir: str, destination: str, shard_indices: list[int]):
    """
    Split a dataset into spans and tokenize, then save each shard to disk.

    Args:
        dataset_dir (str): Directory containing the dataset to process.
        destination (str): Path to save the processed shards.
        shard_indices (list[int]): List of shard indices to process.

    Returns:
        None
    """
    for idx in shard_indices:
        log.info(f"Tokenizing {dataset_dir} shard {idx}...")
        dataset = ds.load_from_disk(dataset_dir)
        shard = dataset.shard(num_shards=100, index=idx)
        tokenized_and_split = shard.map(break_down_into_spans_and_tokenize, batch_size=256, remove_columns=['text'],
                                        batched=True, num_proc=min(os.cpu_count() - 1, 64))
        tokenized_and_split.save_to_disk(f"{destination}/{idx}")


def apply_mlm(dataset_dir: str, destination: str, shard_indices: list[int]):
    """
    Apply Masked Language Modeling (MLM) to a dataset and save to disk.

    Args:
        dataset_dir (str): Directory containing the dataset to process.
        destination (str): Path to save the processed dataset
        shard_indices (list[int]): List of shard indices to process.

    Returns:
        None
    """
    for idx in shard_indices:
        log.info(f"Applying MLM to {dataset_dir} shard {idx}...")
        dataset = ds.load_from_disk(dataset_dir)
        shard = dataset.shard(num_shards=100, index=idx)
        mlm_data = shard.map(mask_batch, batched=True, batch_size=512, num_proc=min(os.cpu_count() - 1, 64))
        mlm_data.save_to_disk(f"{destination}/{idx}")
