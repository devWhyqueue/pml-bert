import os

from datasets import load_dataset, Dataset, load_from_disk

from configuration.config import get_logger, settings

log = get_logger(__name__)

datasets = [
    {
        'identifier': 'wikimedia/wikipedia',
        'short': 'wikipedia',
        'name': '20231101.en',
        'split': 'train'
    },
    {
        'identifier': 'manu/project_gutenberg',
        'short': 'project_gutenberg',
        'name': None,
        'split': 'en'
    },
]


def download_source_datasets() -> list[Dataset]:
    """
    Downloads the specified datasets if they do not already exist.

    For each dataset in the `datasets` list, this function checks if the dataset
    already exists at the specified path. If not, it downloads the dataset and
    saves it to disk. If the dataset already exists, it loads the dataset from disk.

    Returns:
        list[Dataset]: A list of downloaded or loaded datasets.
    """
    downloaded_datasets = []
    for ds in datasets:
        dataset_path = f"{settings['data']['path']}/{ds['short']}"
        if not os.path.exists(dataset_path):
            log.info(f"Downloading {ds['identifier']} dataset...")
            dataset = load_dataset(ds['identifier'], name=ds['name'], split=ds['split'])
            dataset.save_to_disk(dataset_path)
            log.info(f"Saved {ds['identifier']} dataset to {dataset_path}")
        else:
            log.info(f"{ds['identifier']} dataset already exists at {dataset_path}")
            dataset = load_from_disk(dataset_path)

        downloaded_datasets.append(dataset)

    return downloaded_datasets
