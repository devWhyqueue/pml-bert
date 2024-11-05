import datasets as hf_ds
import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, hf_dataset: hf_ds.Dataset):
        self.input_ids = torch.tensor(hf_dataset['input_ids'], dtype=torch.long)
        self.attention_masks = torch.tensor(hf_dataset['attention_mask'], dtype=torch.long)
        self.special_tokens_masks = torch.tensor(hf_dataset['special_tokens_mask'], dtype=torch.long)
        self.mlm_labels = torch.tensor(hf_dataset['mlm_labels'], dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'special_tokens_mask': self.special_tokens_masks[idx],
            'mlm_labels': self.mlm_labels[idx]
        }


def load_data(path: str) -> Dataset:
    """
    Load a dataset from a file.

    Args:
        path (str): The path to the file containing the dataset.

    Returns:
        Dataset: The loaded dataset.
    """
    hf_dataset = hf_ds.load_from_disk(path)
    return TrainingDataset(hf_dataset)
