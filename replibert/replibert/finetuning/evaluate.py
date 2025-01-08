import torch
import torch.distributed as dist
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from configuration.config import get_logger, settings
from data.finetuning.datasets import FineTuningDataset
from model.model import BertToxic, Bert
from utils import is_main_process, _initialize_distributed, get_available_cpus

log = get_logger(__name__)


def evaluate_on_all_datasets(model: nn.Module, train_dataset: FineTuningDataset, val_dataset: FineTuningDataset,
                             test_dataset: FineTuningDataset, config: dict):
    """
    Evaluates the model on the training, validation, and testing datasets.

    Args:
        model (nn.Module): The model to be evaluated.
        train_dataset (FineTuningDataset): The training dataset.
        val_dataset (FineTuningDataset): The validation dataset.
        test_dataset (FineTuningDataset): The testing dataset.
        config (dict): Configuration settings.
    """
    log.info("Evaluating model on training dataset")
    evaluate(train_dataset, model.module.state_dict(), config)
    log.info("Evaluating model on validation dataset")
    evaluate(val_dataset, model.module.state_dict(), config)
    log.info("Evaluating model on testing dataset")
    evaluate(test_dataset, model.module.state_dict(), config)


def evaluate(dataset: FineTuningDataset, weights: str | dict, config: dict = settings["finetuning"]):
    """
    Evaluate the model on the given dataset.

    Args:
        dataset (FineTuningDataset): The dataset to evaluate on.
        weights (str | dict): Path to the model weights or the state dictionary.
        config (dict): Configuration settings.

    Returns:
        None
    """
    rank, world_size = _initialize_distributed()

    test_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    generator = torch.Generator().manual_seed(42)
    test_loader = DataLoader(
        dataset, batch_size=config["batch_size"], sampler=test_sampler,
        num_workers=get_available_cpus(), pin_memory=True, generator=generator
    )

    model = BertToxic(Bert(), num_labels=1)
    state_dict = torch.load(weights, weights_only=True) if isinstance(weights, str) else weights
    model.load_state_dict(state_dict)
    model.to(config["device"])
    model = torch.nn.parallel.DistributedDataParallel(model)

    model.eval()
    local_loss, local_samples, local_labels, local_preds, local_probs \
        = _calculate_local_results(model, test_loader, config['threshold'], config["device"])
    total_loss, total_samples = _aggregate_scalar_values(local_loss, local_samples, config["device"])
    combined_labels, combined_preds, combined_probs = _gather_all_predictions(local_labels, local_preds, local_probs)

    if is_main_process():
        _log_metrics(total_loss, total_samples, combined_labels, combined_preds, combined_probs)


def _calculate_local_results(model: torch.nn.Module, test_loader: DataLoader, threshold: float, device: str):
    """
    Calculate local loss and predictions for the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        threshold (float): Threshold for binary classification.
        device (str): Device to run the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: A tuple containing total loss, total samples, all labels, all predictions, and all probabilities.
    """
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    criterion = nn.BCEWithLogitsLoss().to(device)
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", unit="batch", disable=not is_main_process()):
            input_ids = inputs[:, 0, :].to(device, non_blocking=True)
            attention_mask = inputs[:, 1, :].to(device, non_blocking=True)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).long()
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().round().numpy().astype(int))
            all_probabilities.extend(probabilities.cpu().numpy())

    return total_loss, total_samples, all_labels, all_predictions, all_probabilities


def _aggregate_scalar_values(local_loss: float, local_samples: int, device: str):
    """
    Aggregate scalar values (loss and samples) across processes.

    Args:
        local_loss (float): Local loss value.
        local_samples (int): Local sample count.
        device (str): Device to run the aggregation on (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: A tuple containing total loss and total samples.
    """
    loss_tensor = torch.tensor(local_loss, device=device)
    samples_tensor = torch.tensor(local_samples, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor.item(), samples_tensor.item()


def _gather_all_predictions(local_labels, local_preds, local_probs):
    """
    Gather predictions and labels from all ranks.

    Args:
        local_labels (list): Local labels.
        local_preds (list): Local predictions.
        local_probs (list): Local probabilities.

    Returns:
        tuple: A tuple containing combined labels, combined predictions, and combined probabilities.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    gathered_labels = [None] * world_size
    gathered_predictions = [None] * world_size
    gathered_probabilities = [None] * world_size

    dist.all_gather_object(gathered_labels, local_labels)
    dist.all_gather_object(gathered_predictions, local_preds)
    dist.all_gather_object(gathered_probabilities, local_probs)

    # Flatten gathered lists
    combined_labels = [lbl for rank_labels in gathered_labels for lbl in rank_labels]
    combined_preds = [pred for rank_preds in gathered_predictions for pred in rank_preds]
    combined_probs = [prob for rank_probs in gathered_probabilities for prob in rank_probs]

    return combined_labels, combined_preds, combined_probs


def _log_metrics(total_loss: float, total_samples: int, labels, preds, probs):
    """
    Log evaluation metrics.

    Args:
        total_loss (float): Total loss value.
        total_samples (int): Total sample count.
        labels (list): Combined labels.
        preds (list): Combined predictions.
        probs (list): Combined probabilities.

    Returns:
        None
    """
    avg_loss = total_loss / total_samples
    log.info(f"Evaluation complete - Loss: {avg_loss:.4f}")

    class_report = classification_report(labels, preds)
    log.info(f"Classification Report:\n{class_report}")

    roc_auc = roc_auc_score(labels, probs, average="weighted")
    log.info(f"ROC-AUC Score: {roc_auc:.4f}")
