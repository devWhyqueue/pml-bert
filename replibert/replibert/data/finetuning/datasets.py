from abc import ABC
from typing import Callable, Tuple, Optional

import numpy as np
import torch
from datasets import load_from_disk, Dataset, disable_progress_bar, enable_progress_bar

from utils import get_available_cpus


class FineTuningDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for fine-tuning datasets.

    Args:
        input_fields (list): The input fields to use for the dataset.
        text_field (str): The field name of the text data.
        dataset_dir (str, optional): Directory where the dataset is stored. Defaults to None.
        hf_dataset (optional): Pre-loaded Hugging Face dataset. Defaults to None.
        split (str, optional): Dataset split to use (e.g., 'train', 'test'). Defaults to 'train'.
        transformation (Callable, optional): Transformation function to apply to each example (text). Defaults to None.
    """

    def __init__(self, text_field: str, input_fields: list, label_col: str, dataset_dir: str = None, hf_dataset=None,
                 split: str = 'train', transformation: Optional[Callable] = None):
        self.text_field = text_field
        self.input_fields = input_fields
        self.label_col = label_col
        self.dataset_dir = dataset_dir
        self.split = split
        self.transformation = transformation

        if hf_dataset is not None:
            self.hf_dataset = hf_dataset
        else:
            self.hf_dataset = load_from_disk(dataset_dir)[split]

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.tensor, torch.tensor]: A tuple containing the input tensor and the label tensor.
        """
        item = self.hf_dataset[idx]
        if isinstance(self.input_fields, list):
            xi = torch.tensor([item[field] for field in self.input_fields], dtype=torch.int32)
        elif self.input_fields in item:
            xi = torch.tensor(item[self.input_fields], dtype=torch.float32)
        else:
            xi = item[self.text_field]

        if self.transformation:
            xi = self.transformation(xi)

        yi = self.get_label(item)
        return xi, yi

    def get_texts(self) -> np.ndarray:
        """
        Retrieve all text data from the dataset.

        Returns:
            np.ndarray: Array of text data from the dataset.
        """
        return np.array(self.hf_dataset[self.text_field])

    def get_input_vector(self) -> torch.tensor:
        """
        Abstract method to retrieve the input vector (samples, features) for the dataset.

        Returns:
            torch.tensor: The input vector for the dataset.
        """
        return torch.tensor(self.hf_dataset[self.input_fields], dtype=torch.float32)

    def get_label(self, item: dict) -> torch.tensor:
        """
        Retrieve the label for a given sample.

        Args:
            item: A sample from the dataset.

        Returns:
            Any: The label corresponding to the sample.
        """
        return torch.tensor(item[self.label_col], dtype=torch.float32)

    def get_label_vector(self) -> torch.tensor:
        """
        Retrieve the label vector (samples, labels) for the dataset.

        Returns:
            torch.tensor: The label vector for the dataset.
        """
        return torch.tensor(self.hf_dataset[self.label_col], dtype=torch.float32)


class CivilCommentsDataset(FineTuningDataset):
    """
    Dataset class for Civil Comments data, inheriting from FineTuningDataset.
    """
    def __init__(self, input_fields: list, dataset_dir: str = None, hf_dataset=None, split: str = 'train',
                 transformation: Optional[Callable] = None):
        super().__init__("text", input_fields, "toxicity", dataset_dir, hf_dataset, split, transformation)
        if hf_dataset:
            self.hf_dataset = self._add_column(hf_dataset)
            self.label_col = "max_toxicity"
            self.hf_dataset = self.hf_dataset.select_columns([self.text_field, self.label_col] + self.input_fields)

    @staticmethod
    def _add_column(hf_dataset: Dataset) -> Dataset:
        """
        Adds a column "max_toxicity" to the dataset.
        This column is the result of applying the max operator across the toxicity-related labels.

        Args:
            hf_dataset (Dataset): The Hugging Face dataset to which the column will be added.

        Returns:
            Dataset: The dataset with the added "max_toxicity" column.
        """
        toxicity_labels = ["identity_attack", "insult", "obscene", "severe_toxicity", "sexual_explicit", "threat",
                           "toxicity"]

        def compute_max_toxicity(row: dict) -> float:
            """
            Computes the maximum toxicity value for a given row.
            """
            return max(row[label] for label in toxicity_labels)

        disable_progress_bar()
        hf_dataset = hf_dataset.map(lambda row: {"max_toxicity": compute_max_toxicity(row)},
                                    num_proc=get_available_cpus())
        enable_progress_bar()
        return hf_dataset


class JigsawToxicityDataset(FineTuningDataset):
    """
    Dataset class for Jigsaw Toxicity Prediction data, inheriting from FineTuningDataset.
    """
    def __init__(self, input_fields: list, dataset_dir: str = None, hf_dataset=None, split: str = 'train',
                 transformation: Optional[Callable] = None):
        super().__init__("comment_text", input_fields, "toxic", dataset_dir, hf_dataset, split, transformation)
        if hf_dataset:
            disable_progress_bar()
            hf_dataset = self._filter_rows_with_missing_keys(hf_dataset)
            hf_dataset = self._add_column(hf_dataset)
            enable_progress_bar()
            self.label_col = "any_toxic"
            self.hf_dataset = hf_dataset.select_columns([self.text_field, self.label_col] + self.input_fields)

    @staticmethod
    def _filter_rows_with_missing_keys(hf_dataset: Dataset) -> Dataset:
        """
        Filters out rows where not all required keys are present.
        """
        toxicity_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        def has_all_keys(row):
            return all(label in row for label in toxicity_labels)

        filtered_dataset = hf_dataset.filter(has_all_keys, num_proc=get_available_cpus())
        return filtered_dataset

    @staticmethod
    def _add_column(hf_dataset: Dataset) -> Dataset:
        """
        Adds a column "any_toxic" to the dataset.
        This column is the result of applying logical OR across the toxicity labels.
        """
        toxicity_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        def compute_any_toxic(row: dict) -> float:
            """
            Computes the maximum toxicity value for a given row.
            """
            return float(any(row[label] for label in toxicity_labels))

        hf_dataset = hf_dataset.map(lambda row: {"any_toxic": compute_any_toxic(row)}, num_proc=get_available_cpus())
        return hf_dataset
