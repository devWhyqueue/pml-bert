import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from configuration.config import settings, get_logger
from data.finetuning.datasets import FineTuningDataset
from data.finetuning.transform import balance_dataset
from finetuning.evaluate import evaluate_on_all_datasets
from finetuning.loss import FocalLoss
from model.initialize import initialize_with_weights
from model.model import Bert, BertToxic
from utils import get_available_cpus, is_main_process, _initialize_distributed

log = get_logger(__name__)


def finetune(train_dataset: FineTuningDataset, val_dataset: FineTuningDataset, test_dataset: FineTuningDataset,
             weight_dir: Optional[str] = None, config: dict = settings["finetuning"]) -> None:
    """
    Fine-tunes a BERT model on the training dataset and evaluates it on the validation and testing datasets.

    Args:
        train_dataset (FineTuningDataset): Training dataset.
        val_dataset (FineTuningDataset): Validation dataset.
        test_dataset (FineTuningDataset): Testing dataset.
        weight_dir (Optional[str]): Directory to save the model weights.
        config (dict): Configuration settings.
    """
    rank, world_size = _initialize_distributed()
    log.info(f"Finetuning on {world_size} devices...")

    model = _initialize_model(config["device"])
    model = torch.nn.parallel.DistributedDataParallel(model)
    _finetune_model(model, train_dataset, val_dataset, test_dataset, rank, world_size, config)

    dist.barrier()
    if weight_dir and rank == 0:
        torch.save(model.module.state_dict(), f"{weight_dir}/bert_toxic_weights.pth")
        log.info(f"Model weights saved to {weight_dir}")

    dist.destroy_process_group()


def _initialize_model(device: str) -> BertToxic:
    """
    Initializes the BERT model with pre-trained weights for binary classification.

    Args:
        device (str): The device to perform computation on.

    Returns:
        BertToxic: The initialized BERT model.
    """
    model = Bert()
    initialize_with_weights(model)
    model = BertToxic(model, num_labels=1)
    model.to(device)
    return model


def _finetune_model(model: nn.Module, train_dataset: FineTuningDataset, val_dataset: FineTuningDataset,
                    test_dataset: FineTuningDataset, rank: int, world_size: int, config: dict) -> None:
    """
    Fine-tunes the model on the training data.

    Args:
        model (nn.Module): The model to be fine-tuned.
        train_dataset (FineTuningDataset): The training dataset.
        val_dataset (FineTuningDataset): The validation dataset.
        test_dataset (FineTuningDataset): The testing dataset.
        rank (int): The process rank.
        world_size (int): The total number of processes.
        config (dict): Configuration settings.
    """
    if config["pos_proportion"]:
        train_dataset = balance_dataset(train_dataset, config["pos_proportion"])

    train_loader = _get_train_data_loader(train_dataset, rank, world_size, config)
    optimizer = optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                            weight_decay=float(config["weight_decay"]))
    criterion = FocalLoss().to(config["device"])
    scaler = GradScaler()
    for epoch in range(config["num_epochs"]):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        _finetune_one_epoch(config, criterion, model, optimizer, train_loader, scaler)
        log.info(f"Evaluating trained model after epoch {epoch + 1}")
        evaluate_on_all_datasets(model, train_dataset, val_dataset, test_dataset, config)


def _get_train_data_loader(train_dataset: FineTuningDataset, rank: int, world_size: int, config: dict) -> DataLoader:
    """
    Initializes the training data loader.

    Args:
        train_dataset (FineTuningDataset): Training dataset.
        rank (int): Process rank.
        world_size (int): Total number of processes.
        config (dict, optional): Configuration settings.
    Returns:
        DataLoader: Training data loader.
    """
    generator = torch.Generator().manual_seed(42)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, seed=42)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], sampler=train_sampler, worker_init_fn=_set_seeds,
        num_workers=get_available_cpus(), pin_memory=True, generator=generator
    )

    return train_loader


def _set_seeds(worker_id: int) -> None:
    """
    Sets the random seeds for reproducibility in the worker process.

    Args:
        worker_id (int): The ID of the worker process.
    """
    log.debug(f"Setting seeds for worker {worker_id}")
    torch.use_deterministic_algorithms(True)
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def _finetune_one_epoch(config: dict, criterion: nn.Module, model: nn.Module, optimizer: optim.Optimizer,
                        train_loader: DataLoader, scaler: GradScaler) -> None:
    """
    Performs one epoch of fine-tuning with mixed precision.

    Args:
        config (dict): Configuration settings.
        criterion (nn.Module): Loss function.
        model (nn.Module): BERT model.
        optimizer (optim.Optimizer): Optimizer.
        train_loader (DataLoader): Training data loader.
        scaler (GradScaler): Gradient scaler for mixed precision.
    """
    optimizer.zero_grad()

    for inputs, labels in (
            tqdm(train_loader, desc="Fine-tuning", unit="batch", dynamic_ncols=True, disable=not is_main_process())
    ):
        input_ids, attention_mask, labels = _process_batch(inputs, labels, config)
        with autocast(config["device"]):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.squeeze(-1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()


def _process_batch(inputs: torch.Tensor, labels: torch.Tensor, config: dict):
    """
    Processes a batch and moves data to the appropriate device.

    Args:
        inputs (torch.Tensor): Input tensor.
        labels (torch.Tensor): Labels tensor.
        config (dict): Configuration settings.

    Returns:
        tuple: Processed input IDs, attention masks, and labels.
    """
    input_ids = inputs[:, 0, :].to(config["device"], non_blocking=True)
    attention_mask = inputs[:, 1, :].to(config["device"], non_blocking=True)
    labels = labels.to(config["device"], non_blocking=True)
    return input_ids, attention_mask, labels
