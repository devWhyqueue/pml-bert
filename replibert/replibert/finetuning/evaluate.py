import torch
from torch.utils.data import DataLoader

from configuration.config import get_logger
from model.model import BertToxic

log = get_logger(__name__)


def evaluate(model: torch.nn.Module, test_loader: DataLoader, criterion: torch.nn.modules.loss, device: str):
    """
    Evaluate the performance of the given model on the test dataset.

    Args:
        model (BertToxic): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.modules.loss): Loss function to use for evaluation.
        device (str): Device to run the evaluation on.

    Returns:
        tuple: A tuple containing the average loss and accuracy.
    """
    model.eval()
    total_correct, total_loss, total_samples = _calculate_loss(model, test_loader, criterion, device)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    log.info(f"Evaluation complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


def _calculate_loss(model: torch.nn.Module, test_loader: DataLoader, criterion: torch.nn.modules.loss,
                    device: torch.device):
    """
    Calculate the total loss and accuracy for the given model on the test dataset.

    Args:
        model (BertToxic): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.modules.loss): Loss function to use for evaluation.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: A tuple containing the total correct predictions, total loss, and total samples.
    """
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            input_ids = inputs[:, 0, :].to(device, non_blocking=True)
            attention_mask = inputs[:, 1, :].to(device, non_blocking=True)
            labels = labels.to(device).float()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).long()
            loss = criterion(logits, labels)

            total_correct += (predictions == labels.long()).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_correct, total_loss, total_samples
