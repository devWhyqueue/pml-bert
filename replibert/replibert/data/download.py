import os

from datasets import load_dataset

from configuration.config import get_logger, settings

log = get_logger(__name__)

datasets = [
    {
        'identifier': 'manu/project_gutenberg',
        'short': 'project_gutenberg',
        'name': None,
        'split': 'en'
    },
    {
        'identifier': 'wikimedia/wikipedia',
        'short': 'wikipedia',
        'name': '20231101.en',
        'split': None
    }
]


def download_datasets():
    for ds in datasets:
        dataset_path = f"{settings['data']['path']}/{ds['short']}"
        if not os.path.exists(dataset_path):
            log.info(f"Downloading {ds['identifier']} dataset...")
            dataset = load_dataset(ds['identifier'], name=ds['name'], split=ds['split'])
            dataset.save_to_disk(dataset_path)
            log.info(f"Saved {ds['identifier']} dataset to {dataset_path}")
        else:
            log.info(f"{ds['identifier']} dataset already exists at {dataset_path}")
