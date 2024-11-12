import click
from datasets import load_from_disk

from configuration.config import get_logger, settings
from data.download import download_hf_datasets, HFDataset
from data.training.transform import combine_datasets, spanify_and_tokenize, apply_mlm

log = get_logger(__name__)


@click.group()
def cli():
    """Main CLI for dataset processing."""
    pass


@cli.command()
@click.option("--destination", type=click.Path(), help="Directory where the datasets will be saved.")
@click.option("--train", is_flag=True, help="Flag to indicate downloading training datasets.")
@click.option("--finetuning", is_flag=True, help="Flag to indicate downloading fine-tuning datasets.")
def download(destination: str, train: bool, finetuning: bool):
    """
    Downloads datasets based on the specified options.

    Parameters:
    destination (str): Directory where the datasets will be saved.
    train (bool): Flag to indicate downloading training datasets.
    finetuning (bool): Flag to indicate downloading fine-tuning datasets.

    This function downloads either training or fine-tuning datasets based on the provided flags.
    The datasets are saved to the specified destination directory.
    """
    if train:
        to_be_downloaded = settings['data']['train']
    elif finetuning:
        to_be_downloaded = settings['data']['finetuning']
    else:
        log.error("You must specify either --train or --finetuning.")
        return

    datasets = [HFDataset(**item) for item in to_be_downloaded]
    downloaded_datasets = download_hf_datasets(datasets, destination)
    log.info(f"Downloaded {len(downloaded_datasets)} datasets.")


@cli.command()
@click.option("--dataset_dirs", "-d", type=click.Path(), multiple=True,
              help="Directories containing the datasets to combine.")
@click.option("--destination", type=click.Path(), help="Directory where splits will be saved.")
@click.option("--keep", multiple=True, default=None, help="Columns to keep in the combined dataset.")
@click.option("--shards", default=100, help="Number of shards to split the dataset into.")
@click.option("--shuffle", is_flag=True, help="Shuffle the dataset before splitting.")
def combine(dataset_dirs: tuple[str], destination: str, keep: list[str], shards: int, shuffle: bool):
    """
    Combines, shuffles and saves the specified datasets into shards.

    Parameters:
    dataset_dirs (tuple): Directories containing the datasets to combine
    destination (str): Directory where the combined dataset will be saved.
    keep (list): Columns to keep in the combined dataset. Default is None (keeps all).
    shards (int): Number of shards to split the dataset into. Default is 100.
    shuffle (bool): Shuffle the dataset before splitting. Default is True.

    This function loads the specified datasets and combines them.
    The dataset is saved to the specified destination directory in shards.
    """
    log.info(f"Loading datasets {dataset_dirs}...")
    datasets = [load_from_disk(dataset) for dataset in dataset_dirs]
    combine_datasets(datasets, destination, keep, shards, shuffle)


@cli.command()
@click.option("--dataset_dir", type=click.Path(), help="Directory containing the sharded dataset.")
@click.option("--destination", type=click.Path(), help="Directory where the tokenized dataset will be saved.")
@click.option("--start_index", type=int, help="First shard index to tokenize.")
@click.option("--end_index", type=int, help="Last shard index to tokenize (exclusive).")
def tokenize(dataset_dir: str, destination: str, start_index: int, end_index: int):
    """
    Splits docs into spans and tokenizes specified shard range of the dataset.

    Parameters:
    dataset_dir (str): Directory containing the sharded dataset.
    destination (str): Directory where the tokenized dataset will be saved.
    start_index (int): Starting index of the splits to be tokenized.
    end_index (int): Ending index of the splits to be tokenized (exclusive).

    This function loads the specified dataset shards and tokenizes them.
    The tokenized dataset is saved in the specified destination directory.
    """
    shard_indices = list(range(start_index, end_index))
    spanify_and_tokenize(dataset_dir, destination, shard_indices)
    log.info(f"Tokenized spans generated for shards {shard_indices} in {destination}.")


@cli.command()
@click.option("--dataset_dir", type=click.Path(), help="Directory containing the dataset to process.")
@click.option("--destination", type=click.Path(), help="Directory where the processed dataset will be saved.")
@click.option("--start_index", type=int, help="First shard index to mask.")
@click.option("--end_index", type=int, help="Last shard index to mask (exclusive).")
def mlm(dataset_dir: str, destination: str, start_index: int, end_index: int):
    """
    Apply Masked Language Modeling (MLM) to a dataset and save to disk.

    Parameters:
    dataset_dir (str): Directory containing the dataset to process.
    destination (str): Path to save the processed dataset.
    start_index (int): Starting index of the shards to apply MLM.
    end_index (int): Ending index of the shards to apply MLM (exclusive).

    This function loads the specified dataset and applies Masked Language Modeling (MLM) to it.
    The processed dataset is saved in the specified destination directory.
    """
    shard_indices = list(range(start_index, end_index))
    apply_mlm(dataset_dir, destination, shard_indices)
    log.info(f"MLM applied to {dataset_dir} and saved to {destination}.")


if __name__ == "__main__":
    cli()
