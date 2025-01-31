import click

from configuration.config import get_logger, settings

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
    from data.download import HFDataset, download_hf_datasets

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
    from datasets import load_from_disk
    from data.training.transform import combine_datasets

    log.info(f"Loading datasets {dataset_dirs}...")
    datasets = [load_from_disk(dataset) for dataset in dataset_dirs]
    combine_datasets(datasets, destination, keep, shards, shuffle)


@cli.command()
@click.option("--dataset_dir", type=click.Path(), help="Directory containing the sharded dataset.")
@click.option("--destination", type=click.Path(), help="Directory where the tokenized dataset will be saved.")
@click.option("--start_index", type=int, help="First shard index to tokenize.")
@click.option("--end_index", type=int, help="Last shard index to tokenize (exclusive).")
def spanify(dataset_dir: str, destination: str, start_index: int, end_index: int):
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
    from data.training.transform import spanify_and_tokenize

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
    from data.training.transform import apply_mlm

    shard_indices = list(range(start_index, end_index))
    apply_mlm(dataset_dir, destination, shard_indices)
    log.info(f"MLM applied to {dataset_dir} and saved to {destination}.")


@cli.command()
@click.option(
    "--dataset_name",
    type=click.Choice(['civil_comments', 'jigsaw_toxicity_pred', 'sst2']),
    required=True,
    help="Name of the dataset to use. Options are: 'civil_comments', 'jigsaw_toxicity_pred', 'sst2'."
)
@click.option("--dataset_dir", type=click.Path(exists=True), required=True,
              help="Directory containing the dataset to process.")
@click.option("--dataset_fraction", type=float, default=1.0, help="Fraction of the dataset to use.")
def baseline(dataset_name: str, dataset_dir: str, dataset_fraction: float = 1.0):
    """
    Run a baseline model based on the dataset name.
    Maps dataset name to a specific classification or regression task.
    """
    from data.utils import load_data
    from baseline import binary_classification

    log.info(f"Loading data for {dataset_name}...")
    train_dataset, val_dataset, test_dataset = load_data(dataset_name, dataset_dir, dataset_fraction=dataset_fraction)

    if dataset_name in ['sst2', 'civil_comments','jigsaw_toxicity_pred']:
        log.info("Performing binary classification...")
        binary_classification(train_dataset, val_dataset, test_dataset)
        
    else:
        log.error("Unknown dataset name. This should not happen due to limited options.")


@cli.command()
@click.option("--dataset_dir", type=click.Path(exists=True), required=True,
              help="Directory containing the dataset to process.")
@click.option("--text_field", type=str, required=True, help="Name of the field containing the text data.")
@click.option("--preprocess", is_flag=True, help="Flag to indicate whether to preprocess the text data.")
@click.option("--destination", type=click.Path(), help="Directory where the tokenized dataset will be saved.")
def tokenize(dataset_dir: str, text_field: str, preprocess: bool, destination: str):
    """
    Tokenizes the specified dataset and saves it to the destination directory.

    Parameters:
    dataset_dir (str): Directory containing the dataset to process.
    text_field (str): Name of the field containing the text data.
    destination (str): Directory where the tokenized dataset will be saved.

    This function loads the specified dataset, tokenizes it using BERT, and saves the combined tokenized dataset to the destination directory.
    """
    from datasets import load_from_disk
    from data.finetuning.transform import bert_tokenize
    from data.finetuning.transform import preprocess as preprocess_datasets

    dataset = load_from_disk(dataset_dir)

    if preprocess:
        dataset = preprocess_datasets([dataset], text_field)[0]

    dataset = bert_tokenize(dataset, text_field)
    dataset.save_to_disk(destination)

@cli.command()
@click.option("--dataset_name", type=click.Choice(['civil_comments', 'jigsaw_toxicity_pred', 'sst2']), required=True,
              help="Name of the dataset to use. Options are: 'civil_comments', 'jigsaw_toxicity_pred', 'sst2'.")
@click.option("--dataset_dir", type=click.Path(exists=True), required=True,
              help="Directory containing the tokenized dataset to process.")
@click.option("--weight_dir", type=click.Path(), help="Directory to save the model weights.")
def finetune(dataset_name: str, dataset_dir: str, weight_dir: str = None):
    """
    Fine-tune a BERT model on the specified dataset.

    Parameters:
    dataset_name (str): Name of the dataset to use.
    dataset_dir (str): Directory containing the dataset to process.
    weight_dir (str): Directory to save the model weights.
    """
    from finetuning.classification import finetune as finetune_model
    from data.utils import load_data

    train_dataset, val_dataset, test_dataset = load_data(
        dataset=dataset_name, dataset_dir=dataset_dir, dataset_fraction=settings['finetuning']['dataset_fraction']
    )

    # Assumes tokenization
    train_dataset.input_field = ['input_ids', 'attention_mask']
    val_dataset.input_field = ['input_ids', 'attention_mask']
    test_dataset.input_field = ['input_ids', 'attention_mask']

    finetune_model(train_dataset, val_dataset, test_dataset, weight_dir)


@cli.command()
@click.option("--dataset_name", type=click.Choice(['civil_comments', 'jigsaw_toxicity_pred', 'sst2']), required=True,
              help="Name of the dataset to use. Options are: 'civil_comments', 'jigsaw_toxicity_pred', 'sst2'.")
@click.option("--dataset_dir", type=click.Path(exists=True), required=True,
              help="Directory containing the tokenized dataset to process.")
@click.option("--weight_file", type=click.Path(exists=True), required=True, help="Path to the model weights file.")
def evaluate(dataset_name: str, dataset_dir: str, weight_file: str):
    """
    Evaluate the fine-tuned BERT model on the specified dataset.

    Parameters:
    dataset_name (str): Name of the dataset to use.
    dataset_dir (str): Directory containing the tokenized dataset to process.
    weight_file (str): Path to the model weights file.

    This function loads the specified dataset and evaluates the model using the provided weights.
    """
    from data.utils import load_data
    from finetuning.evaluate import evaluate as evaluate_model

    _, _, test_dataset = load_data(dataset=dataset_name, dataset_dir=dataset_dir)

    # Assumes tokenization
    test_dataset.input_field = ['input_ids', 'attention_mask']

    evaluate_model(test_dataset, weight_file)

@cli.command()
@click.option("--submission_files", type=click.Path(exists=True), required=True, help="Path to the kaggle submission files.")
@click.option("--dataset_split", type=click.Choice(['test', 'validation']), required=True,
              help="Split of the Civil Comments Dataset to use. Options are: 'test', 'validation'")
def evaluate_kaggle(submission_files: str, dataset_split: str):
    """
    Evaluate the fine-tuned BERT model on the specified dataset.

    Parameters:
    submission_files (str): Path to the kaggle submission files.

    This function loads the specified dataset and evaluates the model using the provided submission file.
    """
    from finetuning.evaluate import evaluate_submission


    evaluate_submission(submission_files, dataset_split)

@cli.command()
@click.option("--dataset_name", type=click.Choice(['civil_comments', 'jigsaw_toxicity_pred', 'sst2']), required=True,
    help="Name of the dataset to use. Options are: 'civil_comments', 'jigsaw_toxicity_pred', 'sst2'.")
@click.option("--dataset_dir", type=click.Path(exists=True), required=True,
              help="Directory containing the tokenized dataset to process.")
@click.option(
    "--weights", type=click.Path(exists=True), required=True,help="Path to the trained model for generating explanations.")
@click.option("--save_path", type=click.Path(exists=True), required=True,
              help="Directory to save visualisation of explanation to.")
def explain(dataset_name: str, dataset_dir: str, weights: str, save_path: str):
    """
    Generate explanations for model predictions on a specified dataset.

    Parameters:
    dataset_name (str): Name of the dataset to use.
    dataset_dir (str): Directory containing the dataset to process.
    weights (str): Path to the trained model weights.
    output_path (str): Path to save the generated explanations.
    """

    from data.utils import load_data
    from finetuning.evaluate import explain_false_predictions



    _, _, test_dataset = load_data(dataset=dataset_name, dataset_dir=dataset_dir)

    # Assumes tokenization
    test_dataset.input_field = ['input_ids', 'attention_mask']


    explain_false_predictions(
        weights=weights,
        dataset=test_dataset,
        save_path=save_path
    )


if __name__ == "__main__":
    cli()
