from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional

import numpy as np
import torch
from datasets import load_from_disk, concatenate_datasets


class FineTuningDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for fine-tuning datasets.

    Args:
        input_field (str): The field name of the vectorized input data.
        text_field (str): The field name of the text data.
        dataset_dir (str, optional): Directory where the dataset is stored. Defaults to None.
        hf_dataset (optional): Pre-loaded Hugging Face dataset. Defaults to None.
        split (str, optional): Dataset split to use (e.g., 'train', 'test'). Defaults to 'train'.
        n_samples (int, optional): Number of samples to use from the dataset. Defaults to None.
        transformation (Callable, optional): Transformation function to apply to each example (text). Defaults to None.
    """

    def __init__(self, input_field: str, text_field: str, dataset_dir: str = None, hf_dataset=None,
                 split: str = 'train', n_samples: Optional[int] = None, transformation: Optional[Callable] = None):
        self.input_field = input_field
        self.text_field = text_field
        self.dataset_dir = dataset_dir
        self.split = split
        self.n_samples = n_samples
        self.transformation = transformation

        if hf_dataset is not None:
            self.hf_dataset = hf_dataset
        else:
            self._load_hf_dataset(dataset_dir, split)
            if n_samples is not None:
                self.hf_dataset = self.hf_dataset.select(range(min(n_samples, len(self.hf_dataset))))

    def _load_hf_dataset(self, dataset_dir: str, split: str):
        """
        Load the Hugging Face dataset from disk.

        Args:
            dataset_dir (str): Directory where the dataset is stored.
            split (str): Dataset split to use (e.g., 'train', 'test').

        Sets:
            self.hf_dataset: Loaded dataset split.
        """
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
        if self.input_field in item:
            xi = torch.tensor(item[self.input_field], dtype=torch.float32)
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

    @abstractmethod
    def get_label(self, item: dict) -> torch.tensor:
        """
        Abstract method to retrieve the label for a given sample.

        Args:
            item: A sample from the dataset.

        Returns:
            Any: The label corresponding to the sample.
        """
        pass

    @abstractmethod
    def get_label_vector(self) -> torch.tensor:
        """
        Abstract method to retrieve the label vector (samples, labels) for the dataset.

        Returns:
            torch.tensor: The label vector for the dataset.
        """
        pass


class CivilCommentsDataset(FineTuningDataset):
    def __init__(self, input_field: str = "tf_idf", dataset_dir: str = None, hf_dataset=None, split: str = 'train',
                 n_samples: Optional[int] = None, transformation: Optional[Callable] = None):
        super().__init__(input_field, "text", dataset_dir, hf_dataset, split, n_samples, transformation)

    def _load_hf_dataset(self, dataset_dir: str, split: str):
        if split == 'train':
            dataset = load_from_disk(dataset_dir)
            self.hf_dataset = concatenate_datasets([dataset['train'], dataset['validation']])
        else:
            super()._load_hf_dataset(dataset_dir, split)

    def get_label(self, item: dict) -> torch.tensor:
        return torch.tensor(item["toxicity"], dtype=torch.float32)

    def get_label_vector(self) -> torch.tensor:
        return torch.tensor(self.hf_dataset["toxicity"], dtype=torch.float32)


class JigsawToxicityDataset(FineTuningDataset):
    def __init__(self, input_field: str = "tf_idf", dataset_dir: str = None, hf_dataset=None, split: str = 'train',
                 n_samples: Optional[int] = None, transformation: Optional[Callable] = None):
        super().__init__(input_field, "comment_text", dataset_dir, hf_dataset, split, n_samples, transformation)

    def get_label(self, item: dict) -> torch.tensor:
        yi = torch.tensor([
            item["toxic"],
            item["severe_toxic"],
            item["obscene"],
            item["threat"],
            item["insult"],
            item["identity_hate"]
        ], dtype=torch.int8)
        return yi

    def get_label_vector(self) -> torch.tensor:
        y = torch.tensor([
            self.hf_dataset["toxic"],
            self.hf_dataset["severe_toxic"],
            self.hf_dataset["obscene"],
            self.hf_dataset["threat"],
            self.hf_dataset["insult"],
            self.hf_dataset["identity_hate"]
        ], dtype=torch.int8)
        return y.T


class SST2Dataset(FineTuningDataset):
    def __init__(self, input_field: str = "tf_idf", dataset_dir: str = None, hf_dataset=None, split: str = 'train',
                 n_samples: Optional[int] = None, transformation: Optional[Callable] = None):
        super().__init__(input_field, "sentence", dataset_dir, hf_dataset, split, n_samples, transformation)

    def get_label(self, item: dict) -> torch.tensor:
        yi = torch.tensor(item["label"], dtype=torch.int8)
        return yi

    def get_label_vector(self) -> torch.tensor:
        return torch.tensor(self.hf_dataset["label"], dtype=torch.int8)
