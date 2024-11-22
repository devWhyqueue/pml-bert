from typing import Callable, Tuple, Optional, Any, Type

import torch

from configuration.config import get_logger
from data.finetuning.datasets import CivilCommentsDataset, JigsawToxicityDataset, SST2Dataset

log = get_logger(__name__)


def load_data(
        dataset: str,
        dataset_dir: str,
        transformation: Optional[Callable[[Any], Any]] = None,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """


    Args:
        dataset (str): The name of the dataset to load ('civil_comments', 'jigsaw_toxicity_pred', 'sst2').
        dataset_dir (str): Directory where the dataset is stored.
        transformation (Callable[[Any], Any], optional): Transformation function to apply to each sample.
        n_train (Optional[int], optional): Number of training samples.
        n_test (Optional[int], optional): Number of test samples.

    Returns:

    """
    dataset_class = _get_dataset_class(dataset)
    return _create_train_test_datasets(dataset_class, dataset_dir, transformation, n_train, n_test)


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


def _create_train_test_datasets(dataset_class: Type, dataset_dir: str, transformation: Optional[Callable],
                                n_train: Optional[int], n_test: Optional[int]) \
        -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Create training and test datasets.

    Args:
        dataset_class (Type): The class of the dataset.
        dataset_dir (str): Directory where the dataset is stored.
        transformation (Optional[Callable]): Transformation function to apply to each sample.
        n_train (Optional[int]): Number of training samples.
        n_test (Optional[int]): Number of test samples.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Training and test datasets.
    """
    train_dataset = dataset_class(dataset_dir=dataset_dir, split='train', n_samples=n_train,
                                  transformation=transformation)
    try:
        # Labels are hidden in SST2 test set
        test_split = 'validation' if dataset_class == SST2Dataset else 'test'
        test_dataset = dataset_class(dataset_dir=dataset_dir, split=test_split, n_samples=n_test,
                                     transformation=transformation)
    except KeyError:
        test_size_ratio = _get_test_size_ratio(n_test, len(train_dataset))
        split = train_dataset.hf_dataset.train_test_split(test_size=test_size_ratio)
        train_dataset.hf_dataset = split['train']
        test_dataset = dataset_class(hf_dataset=split['test'], transformation=transformation)

    return train_dataset, test_dataset


def _get_test_size_ratio(n_test: Optional[int], train_len: int) -> float:
    """
    Calculate the test size ratio.

    Args:
        n_test (Optional[int]): Number of test samples.
        train_len (int): Length of the training dataset.

    Returns:
        float: The ratio of test samples to training samples.
    """
    return n_test / train_len if n_test is not None and isinstance(n_test, int) else 0.2
