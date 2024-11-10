import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from datasets import load_dataset, Dataset, load_from_disk

from configuration.config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class HFDataset:
    identifier: str
    short: str
    name: Optional[str] = None
    split: Optional[str] = None
    config_kwargs: Optional[Dict[str, Any]] = None


def download_hf_datasets(datasets: list[HFDataset], destination: str) -> list[Dataset]:
    """
    Downloads the specified datasets if they do not already exist.

    For each dataset in the `datasets` list, this function checks if the dataset
    already exists at the specified path. If not, it downloads the dataset and
    saves it to disk. If the dataset already exists, it loads the dataset from disk.

    Args:
        datasets (list[HFDataset]): A list of HFDataset objects specifying the datasets to download.
        destination (str): The directory where the datasets should be saved.

    Returns:
        list[Dataset]: A list of downloaded or loaded datasets.
    """
    downloaded_datasets = []
    for ds in datasets:
        dataset_path = f"{destination}/{ds.short}"
        if not os.path.exists(dataset_path):
            log.info(f"Downloading {ds.identifier} dataset...")
            dataset = load_dataset(ds.identifier, name=ds.name, split=ds.split,
                                   trust_remote_code=True, **(ds.config_kwargs or {}))
            dataset.save_to_disk(dataset_path)
            log.info(f"Saved {ds.identifier} dataset to {dataset_path}")
        else:
            log.info(f"{ds.identifier} dataset already exists at {dataset_path}")
            dataset = load_from_disk(dataset_path)

        downloaded_datasets.append(dataset)

    return downloaded_datasets
