import click
from datasets import load_from_disk

from configuration.config import get_logger
from data.download import download_source_datasets
from data.transform import combine_datasets, spanify_and_tokenize

log = get_logger(__name__)


@click.group()
def cli():
    """Main CLI for dataset processing."""
    pass


@cli.command()
def download():
    """Download source datasets."""
    datasets = download_source_datasets()
    log.info(f"Downloaded {len(datasets)} datasets.")


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


if __name__ == "__main__":
    cli()
