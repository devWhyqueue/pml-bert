from typing import Callable, Tuple, Optional, Any, Type

from datasets import Dataset, ClassLabel, disable_progress_bar, enable_progress_bar

from configuration.config import get_logger
from data.finetuning.datasets import CivilCommentsDataset, JigsawToxicityDataset, SST2Dataset, FineTuningDataset
from utils import get_available_cpus

log = get_logger(__name__)


def load_data(
        dataset: str,
        dataset_dir: str,
        transformation: Optional[Callable[[Any], Any]] = None,
        dataset_fraction: float = 1.0
) -> Tuple[FineTuningDataset, FineTuningDataset, FineTuningDataset]:
    """
    Load train, validation, and test datasets for a given dataset.

    Args:
        dataset (str): The name of the dataset to load ('civil_comments', 'jigsaw_toxicity_pred', 'sst2').
        dataset_dir (str): Directory where the dataset is stored.
        transformation (Callable[[Any], Any], optional): Transformation function to apply to each sample.
        dataset_fraction (float, optional): Fraction of each split to use. For example, 0.5 means use half of each split.

    Returns:
        Tuple[FineTuningDataset, FineTuningDataset, FineTuningDataset]: Training, validation, and test datasets.
    """
    dataset_class = _get_dataset_class(dataset)

    if dataset == 'civil_comments':
        # Has train, validation, test splits directly
        train_dataset = dataset_class(dataset_dir=dataset_dir, split='train', transformation=transformation)
        val_dataset = dataset_class(dataset_dir=dataset_dir, split='validation', transformation=transformation)
        test_dataset = dataset_class(dataset_dir=dataset_dir, split='test', transformation=transformation)

    elif dataset == 'jigsaw_toxicity_pred':
        # Has train and test only, no val split
        train_dataset = dataset_class(dataset_dir=dataset_dir, split='train', transformation=transformation)
        test_dataset = dataset_class(dataset_dir=dataset_dir, split='test', transformation=transformation)

        # Create val split from train
        train_dataset, val_dataset = _split_dataset(
            dataset_class, train_dataset, test_size=0.2, transformation=transformation
        )

    elif dataset == 'sst2':
        # Has train, validation, and test, but test is not usable
        train_dataset = dataset_class(dataset_dir=dataset_dir, split='train', transformation=transformation)
        val_dataset = dataset_class(dataset_dir=dataset_dir, split='validation', transformation=transformation)

        # Discard official test, create our own test from train
        train_dataset, test_dataset = _split_dataset(
            dataset_class, train_dataset, test_size=0.2, transformation=transformation
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Now apply dataset_size fraction to each split stratified by label
    train_dataset = _subset_dataset(dataset_class, train_dataset, dataset_fraction, transformation=transformation)
    val_dataset = _subset_dataset(dataset_class, val_dataset, dataset_fraction, transformation=transformation)
    test_dataset = _subset_dataset(dataset_class, test_dataset, dataset_fraction, transformation=transformation)

    return train_dataset, val_dataset, test_dataset


def _get_dataset_class(dataset: str) -> Type:
    """
    Get the dataset class based on the dataset name.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        Type: The dataset class corresponding to the dataset name.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    dataset_classes = {
        'civil_comments': CivilCommentsDataset,
        'jigsaw_toxicity_pred': JigsawToxicityDataset,
        'sst2': SST2Dataset,
    }
    if dataset not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dataset_classes[dataset]


def _split_dataset(dataset_class: Type,
                   dataset: FineTuningDataset,
                   test_size: float,
                   transformation: Optional[Callable]) -> Tuple[FineTuningDataset, FineTuningDataset]:
    """
    Split a dataset into two subsets using stratification by label.
    `test_size` is a fraction of the dataset to become the second subset.

    Returns: (subset1, subset2)
    """
    total_len = len(dataset)
    actual_test_size = max(1, int(total_len * test_size))
    train_split, test_split = _prepare_split(dataset, dataset.label_col, actual_test_size)
    subset1 = dataset_class(hf_dataset=train_split, transformation=transformation)
    subset2 = dataset_class(hf_dataset=test_split, transformation=transformation)

    return subset1, subset2


def _prepare_split(dataset: FineTuningDataset, label_col: str, split_size: float) -> Tuple[Dataset, Dataset]:
    """
    Prepare a stratified split of the dataset.

    Args:
        dataset (FineTuningDataset): The dataset to split.
        label_col (str): The label column for stratification.
        split_size (float): The size of the split (fraction or count).

    Returns:
        Dataset: The resulting split dataset.
    """
    disable_progress_bar()
    dataset.hf_dataset, label_col = _class_encode_column(dataset.hf_dataset, label_col)
    split = dataset.hf_dataset.train_test_split(
        test_size=split_size, stratify_by_column=label_col, seed=42
    )

    if label_col == "binary_label":
        split.remove_columns(["binary_label"])

    enable_progress_bar()
    return split['train'], split['test']


def _class_encode_column(dataset: Dataset, label_col: str) -> Tuple[Dataset, str]:
    """
    Encode the label column of the dataset for stratification.

    If the label column contains float values, it will be binarized to integer values
    for stratification purposes. The function will then encode the label column.

    Args:
        dataset (Dataset): The dataset containing the label column.
        label_col (str): The name of the label column.

    Returns:
        Tuple[Dataset, str]: The modified dataset and the name of the encoded label column.
    """
    if isinstance(dataset[label_col][0], float):
        def _binarize_labels(example: dict) -> dict:
            example["binary_label"] = 1 if example[label_col] > 0.5 else 0
            return example

        dataset = dataset.map(_binarize_labels, num_proc=get_available_cpus())
        label_col = "binary_label"

    if not isinstance(dataset.features[label_col], ClassLabel):
        dataset = dataset.class_encode_column(label_col)

    return dataset, label_col


def _subset_dataset(dataset_class: Type,
                    dataset: FineTuningDataset,
                    dataset_size: float,
                    transformation: Optional[Callable]) -> FineTuningDataset:
    """
    Reduce the dataset to the given fraction of its original size in a stratified way.
    If dataset_size >= 1.0, returns the dataset unchanged.
    """
    if dataset_size >= 1.0:
        return dataset

    total_len = len(dataset)
    new_len = max(1, int(total_len * dataset_size))
    reduced_split, _ = _prepare_split(dataset, dataset.label_col, (total_len - new_len))
    reduced_dataset = dataset_class(hf_dataset=reduced_split, transformation=transformation)

    return reduced_dataset
