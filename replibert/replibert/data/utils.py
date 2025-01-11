from typing import Callable, Tuple, Optional, Any, Type

from datasets import Dataset, disable_progress_bar, enable_progress_bar, ClassLabel

from configuration.config import get_logger
from data.finetuning.datasets import CivilCommentsDataset, JigsawToxicityDataset, FineTuningDataset
from utils import get_available_cpus

log = get_logger(__name__)


def load_data(dataset: str, input_fields: list, dataset_dir: str, transformation: Optional[Callable[[Any], Any]] = None,
              dataset_fraction: float = 1.0) -> Tuple[FineTuningDataset, FineTuningDataset, FineTuningDataset]:
    """
    Load train, validation, and test datasets for a given dataset.

    Args:
        dataset (str): The name of the dataset to load ('civil_comments', 'jigsaw_toxicity_pred').
        input_fields (list): The input fields to use for the dataset.
        dataset_dir (str): Directory where the dataset is stored.
        transformation (Callable[[Any], Any], optional): Transformation function to apply to each sample.
        dataset_fraction (float, optional): Fraction of each split to use. For example, 0.5 means use half of each split.

    Returns:
        Tuple[FineTuningDataset, FineTuningDataset, FineTuningDataset]: Training, validation, and test datasets.
    """
    dataset_class = _get_dataset_class(dataset)

    if dataset == 'civil_comments':
        # Has train, validation, test splits directly
        train_dataset = dataset_class(input_fields=input_fields, dataset_dir=dataset_dir, split='train',
                                      transformation=transformation)
        val_dataset = dataset_class(input_fields=input_fields, dataset_dir=dataset_dir, split='validation',
                                    transformation=transformation)
        test_dataset = dataset_class(input_fields=input_fields, dataset_dir=dataset_dir, split='test',
                                     transformation=transformation)
    elif dataset == 'jigsaw_toxicity_pred':
        # Has train and test only, no val split
        train_dataset = dataset_class(input_fields=input_fields, dataset_dir=dataset_dir, split='train',
                                      transformation=transformation)
        test_dataset = dataset_class(input_fields=input_fields, dataset_dir=dataset_dir, split='test',
                                     transformation=transformation)

        # Create val split from train
        train_dataset, val_dataset = _split_dataset(
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
        'jigsaw_toxicity_pred': JigsawToxicityDataset
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
    subset1 = dataset_class(input_fields=dataset.input_fields, hf_dataset=train_split, transformation=transformation)
    subset2 = dataset_class(input_fields=dataset.input_fields, hf_dataset=test_split, transformation=transformation)

    return subset1, subset2


def _prepare_split(dataset: FineTuningDataset, label_col: str, test_size: float = None) -> Tuple[Dataset, Dataset]:
    """
    Prepare a stratified split of the dataset.

    Args:
        dataset (FineTuningDataset): The dataset to split.
        label_col (str): The label column for stratification.
        test_size (float): The test split size.

    Returns:
        Dataset: The resulting split dataset.
    """
    disable_progress_bar()
    dataset.hf_dataset = _class_encode_column(dataset.hf_dataset, label_col)

    if test_size is not None:
        split = dataset.hf_dataset.train_test_split(
            test_size=test_size, stratify_by_column=label_col, seed=42
        )
        split = (split["train"], split["test"])
    else:
        split = (dataset.hf_dataset, None)

    enable_progress_bar()
    return split


def _class_encode_column(dataset: Dataset, label_col: str) -> Dataset:
    def _binarize_labels(example: dict) -> dict:
        example[label_col] = 1.0 if example[label_col] > 0.5 else 0.0
        return example

    dataset = dataset.map(_binarize_labels, num_proc=get_available_cpus())
    if not isinstance(dataset.features[label_col], ClassLabel):
        dataset = dataset.class_encode_column(label_col)
    dataset = dataset.cast_column(label_col, ClassLabel(names=['false', 'true']))

    return dataset


def _subset_dataset(dataset_class: Type,
                    dataset: FineTuningDataset,
                    dataset_size: float,
                    transformation: Optional[Callable]) -> FineTuningDataset:
    """
    Reduce the dataset to the given fraction of its original size in a stratified way.
    If dataset_size >= 1.0, returns the dataset unchanged.
    """
    if dataset_size >= 1.0:
        test_size = None
    else:
        total_len = len(dataset)
        new_len = max(1, int(total_len * dataset_size))
        test_size = total_len - new_len

    reduced_split, _ = _prepare_split(dataset, dataset.label_col, test_size)
    reduced_dataset = dataset_class(input_fields=dataset.input_fields, hf_dataset=reduced_split,
                                    transformation=transformation)

    return reduced_dataset
