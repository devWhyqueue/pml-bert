from typing import Callable, Tuple, Optional, Any

import torch
from datasets import load_from_disk


class FineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str = None, split: str = 'train', n_samples: Optional[int] = None,
                 transformation: Optional[Callable] = None, hf_dataset=None):
        if hf_dataset is not None:
            self.hf_dataset = hf_dataset
        else:
            self.hf_dataset = load_from_disk(dataset_dir)[split]
            if n_samples is not None:
                self.hf_dataset = self.hf_dataset.select(range(min(n_samples, len(self.hf_dataset))))
        self.transformation = transformation

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        raise NotImplementedError("Subclasses must implement this method")


class CivilCommentsDataset(FineTuningDataset):
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        item = self.hf_dataset[idx]
        xi = item["text"]
        yi = {
            "identity_attack": torch.tensor(item["identity_attack"], dtype=torch.float),
            "insult": torch.tensor(item["insult"], dtype=torch.float),
            "obscene": torch.tensor(item["obscene"], dtype=torch.float),
            "severe_toxicity": torch.tensor(item["severe_toxicity"], dtype=torch.float),
            "sexual_explicit": torch.tensor(item["sexual_explicit"], dtype=torch.float),
            "threat": torch.tensor(item["threat"], dtype=torch.float),
            "toxicity": torch.tensor(item["toxicity"], dtype=torch.float)
        }
        if self.transformation:
            xi = self.transformation(xi)
        return xi, yi


class JigsawToxicityDataset(FineTuningDataset):
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        item = self.hf_dataset[idx]
        xi = item["comment_text"]
        yi = {
            "toxic": torch.tensor(item["toxicity"], dtype=torch.float),
            "severe_toxic": torch.tensor(item["severe_toxicity"], dtype=torch.float),
            "obscene": torch.tensor(item["obscene"], dtype=torch.float),
            "threat": torch.tensor(item["threat"], dtype=torch.float),
            "insult": torch.tensor(item["insult"], dtype=torch.float),
            "identity_hate": torch.tensor(item["identity_attack"], dtype=torch.float)
        }
        if self.transformation:
            xi = self.transformation(xi)
        return xi, yi


class SST2Dataset(FineTuningDataset):
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        item = self.hf_dataset[idx]
        xi = item["sentence"]
        yi = torch.tensor(item["label"], dtype=torch.long)
        if self.transformation:
            xi = self.transformation(xi)
        return xi, yi
