from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from configuration.config import settings, get_logger
from data.finetuning.datasets import FineTuningDataset
from finetuning.evaluate import evaluate
from model.initialize import initialize_with_weights
from model.model import Bert, BertToxic
from utils import get_available_cpus, is_main_process

log = get_logger(__name__)


def finetune(train_dataset: FineTuningDataset, test_dataset: FineTuningDataset, weights_dir: Optional[str] = None,
             config: dict = settings["finetuning"]) -> None:
    """
    Fine-tunes a BERT model on the specified training dataset and evaluates it on the testing dataset.

    Args:
        train_dataset (FineTuningDataset): Training dataset.
        test_dataset (FineTuningDataset): Testing dataset.
        weights_dir (Optional[str]): Directory to save the model weights.
        config (dict): Configuration settings.
    """
    rank, world_size = _initialize_distributed()
    log.info(f"Finetuning on {world_size} devices...")

    log.info("Preparing data loaders...")
    train_loader, test_loader = _get_data_loader(train_dataset, test_dataset, rank, world_size, config)
    model = _initialize_model(config["device"])
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer, criterion, scaler = _initialize_training_components(model, config)
    _finetune_model(model, train_loader, optimizer, criterion, scaler, config)

    _evaluate_model(model, test_loader, criterion, config["device"])

    if weights_dir and rank == 0:
        _save_model_weights(model, weights_dir)

    dist.destroy_process_group()


def _initialize_distributed() -> Tuple[int, int]:
    """
    Initializes the distributed environment.

    Returns:
        Tuple[int, int]: The rank and world size.
    """
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return rank, world_size


def _get_data_loader(train_dataset: FineTuningDataset, test_dataset: FineTuningDataset, rank: int, world_size: int,
                     config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the training and testing data loaders with DistributedSampler.

    Args:
        train_dataset (FineTuningDataset): Training dataset.
        test_dataset (FineTuningDataset): Testing dataset.
        rank (int): Process rank.
        world_size (int): Total number of processes.
        config (dict, optional): Configuration settings.

    Returns:
        tuple: Training and testing data loaders.
    """
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], sampler=train_sampler,
        num_workers=get_available_cpus(), pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], sampler=test_sampler,
        num_workers=get_available_cpus(), pin_memory=True
    )

    return train_loader, test_loader


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


def _initialize_training_components(model: nn.Module, config: dict) -> Tuple[optim.Optimizer, nn.Module, GradScaler]:
    """
    Initializes the optimizer, criterion, and gradient scaler.

    Args:
        model (nn.Module): The model to be trained.
        config (dict): Configuration settings.

    Returns:
        Tuple[optim.Optimizer, nn.Module, GradScaler]: The optimizer, criterion, and gradient scaler.
    """
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    criterion = nn.BCEWithLogitsLoss().to(config["device"])
    scaler = GradScaler()
    return optimizer, criterion, scaler


def _finetune_model(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module,
                    scaler: GradScaler, config: dict) -> None:
    """
    Fine-tunes the model on the training data.

    Args:
        model (nn.Module): The model to be fine-tuned.
        train_loader (DataLoader): The training data loader.
        optimizer (optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        scaler (GradScaler): The gradient scaler.
        config (dict): Configuration settings.
    """
    for epoch in range(config["num_epochs"]):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        total_loss, total_samples = _finetune_one_epoch(config, criterion, model, optimizer, train_loader, scaler)
        avg_loss = total_loss / total_samples
        log.info(f'Epoch {epoch + 1}/{config["num_epochs"]} - Loss: {avg_loss:.4f}')


def _save_model_weights(model: nn.Module, weights_dir: str) -> None:
    """
    Saves the model weights.

    Args:
        model (nn.Module): The model to be saved.
        weights_dir (str): The directory to save the model weights.
    """
    log.info("Saving model weights")
    torch.save(model.module.state_dict(), f"{weights_dir}/bert_toxic_weights.pth")


def _evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: str) -> None:
    """
    Evaluates the model.

    Args:
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): The test data loader.
        criterion (nn.Module): The loss function.
        device (str): The device to perform computation on.
    """
    log.info("Evaluating trained model")
    evaluate(model.module, test_loader, criterion, device)


def _finetune_one_epoch(config: dict, criterion: nn.Module, model: nn.Module, optimizer: optim.Optimizer,
                        train_loader: DataLoader, scaler: GradScaler, log_interval: int = 1_000) -> Tuple[float, int]:
    """
    Performs one epoch of fine-tuning with mixed precision.

    Args:
        config (dict): Configuration settings.
        criterion (nn.Module): Loss function.
        model (nn.Module): BERT model.
        optimizer (optim.Optimizer): Optimizer.
        train_loader (DataLoader): Training data loader.
        scaler (GradScaler): Gradient scaler for mixed precision.
        log_interval (int): Logging interval.

    Returns:
        tuple: Total loss and total samples processed.
    """
    total_loss = 0.0
    total_samples = 0
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(
            tqdm(train_loader, desc="Fine-tuning", unit="batch", dynamic_ncols=True, disable=not is_main_process())
    ):
        input_ids, attention_mask, labels = _process_batch(inputs, labels, config)

        optimizer.zero_grad()
        with autocast(str(config["device"])):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.squeeze(-1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        running_loss += loss.item()
        total_samples += batch_size

        if (batch_idx + 1) % log_interval == 0:
            running_loss = _log_progress(batch_idx, log_interval, running_loss, train_loader)

    return total_loss, total_samples


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


def _compute_loss(model: nn.Module, criterion: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  labels: torch.Tensor, device: str) -> torch.Tensor:
    """
    Computes the loss using mixed precision.

    Args:
        model (nn.Module): The BERT model.
        criterion (nn.Module): The loss function.
        input_ids (torch.Tensor): Tensor of input IDs.
        attention_mask (torch.Tensor): Tensor of attention masks.
        labels (torch.Tensor): Tensor of labels.
        device (str): The device to perform computation on.

    Returns:
        torch.Tensor: The computed loss.
    """
    with autocast(device):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
        return criterion(logits, labels)


def _log_progress(batch_idx: int, log_interval: int, running_loss: float, train_loader: DataLoader) -> float:
    """
    Logs progress at specified intervals and resets the running loss.

    Args:
        batch_idx (int): The current batch index.
        log_interval (int): The interval at which to log progress.
        running_loss (float): The running loss.
        train_loader (DataLoader): The training data loader.

    Returns:
        float: The reset running loss (always 0).
    """
    avg_loss = running_loss / log_interval
    log.info(f"[Batch {batch_idx + 1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
    return 0
