from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from configuration.config import settings, get_logger
from data.finetuning.datasets import FineTuningDataset
from finetuning.evaluate import evaluate_on_all_datasets
from model.initialize import initialize_with_weights
from model.model import Bert, BertToxic
from utils import get_available_cpus, is_main_process, _initialize_distributed

log = get_logger(__name__)


def finetune(train_dataset: FineTuningDataset, val_dataset: FineTuningDataset, test_dataset: FineTuningDataset,
             weights_dir: Optional[str] = None, config: dict = settings["finetuning"]) -> None:
    """
    Fine-tunes a BERT model on the training dataset and evaluates it on the validation and testing datasets.

    Args:
        train_dataset (FineTuningDataset): Training dataset.
        val_dataset (FineTuningDataset): Validation dataset.
        test_dataset (FineTuningDataset): Testing dataset.
        weights_dir (Optional[str]): Directory to save the model weights.
        config (dict): Configuration settings.
    """
    rank, world_size = _initialize_distributed()
    log.info(f"Finetuning on {world_size} devices...")

    model = _initialize_model(config["device"])
    model = torch.nn.parallel.DistributedDataParallel(model)
    _finetune_model(model, train_dataset, val_dataset, test_dataset, rank, world_size, config)

    dist.barrier()
    if weights_dir and rank == 0:
        log.info("Saving model weights")
        torch.save(model.module.state_dict(), f"{weights_dir}/bert_toxic_weights.pth")

    dist.destroy_process_group()


def _get_data_loader(train_dataset: FineTuningDataset, rank: int, world_size: int, config: dict) -> DataLoader:
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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], sampler=train_sampler,
        num_workers=get_available_cpus(), pin_memory=True
    )

    return train_loader


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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], sampler=train_sampler,
        num_workers=get_available_cpus(), pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                           weight_decay=float(config["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    criterion = nn.BCEWithLogitsLoss().to(config["device"])
    scaler = GradScaler()
    for epoch in range(config["num_epochs"]):
        log.info(f"Learning rate in epoch {epoch + 1} is {scheduler.get_last_lr()[0]:.6f}")
        train_loader.sampler.set_epoch(epoch)
        model.train()
        total_loss, total_samples = _finetune_one_epoch(config, criterion, model, optimizer, train_loader, scaler)
        avg_loss = total_loss / total_samples
        log.info(f"Evaluating trained model after epoch {epoch + 1}")
        log.info(f'Epoch {epoch + 1}/{config["num_epochs"]} - Loss: {avg_loss:.4f}')
        evaluate_on_all_datasets(model, train_dataset, val_dataset, test_dataset, config)
        scheduler.step()


def _finetune_one_epoch(config: dict, criterion: nn.Module, model: nn.Module, optimizer: optim.Optimizer,
                        train_loader: DataLoader, scaler: GradScaler) -> Tuple[float, int]:
    """
    Performs one epoch of fine-tuning with mixed precision and gradient accumulation.

    Args:
        config (dict): Configuration settings.
        criterion (nn.Module): Loss function.
        model (nn.Module): BERT model.
        optimizer (optim.Optimizer): Optimizer.
        train_loader (DataLoader): Training data loader.
        scaler (GradScaler): Gradient scaler for mixed precision.

    Returns:
        tuple: Total loss and total samples processed.
    """
    total_loss, total_samples = 0.0, 0
    actual_batch_size = 16  # Assumed max batch size fitting into memory
    accumulation_steps = config["batch_size"] // actual_batch_size
    optimizer.zero_grad()

    for batch_idx, (inputs, labels) in enumerate(
            tqdm(train_loader, desc="Fine-tuning", unit="batch", dynamic_ncols=True, disable=not is_main_process())
    ):
        input_ids, attention_mask, labels = _process_batch(inputs, labels, config)
        loss = _process_batch_with_autocast(model, criterion, config, input_ids,
                                            attention_mask, labels, accumulation_steps)

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps  # Reverse normalization
        total_samples += batch_size

    return total_loss, total_samples


def _process_batch_with_autocast(model: nn.Module, criterion: nn.Module, config: dict,
                                 input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
                                 accumulation_steps: int) -> torch.Tensor:
    """
    Process a single batch with mixed precision, computing the loss.

    Args:
        model (nn.Module): The BERT model.
        criterion (nn.Module): Loss function.
        config (dict): Configuration settings.
        input_ids (torch.Tensor): Input IDs tensor.
        attention_mask (torch.Tensor): Attention mask tensor.
        labels (torch.Tensor): Target labels tensor.
        accumulation_steps (int): Number of accumulation steps.

    Returns:
        torch.Tensor: Scaled and normalized loss.
    """
    with autocast(config["device"]):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.squeeze(-1)
        loss = criterion(logits, labels)
        return loss / accumulation_steps


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
