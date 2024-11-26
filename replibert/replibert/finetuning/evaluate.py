import torch
from torch.utils.data import DataLoader

from configuration.config import get_logger
from model.model import BertToxic

log = get_logger(__name__)


def evaluate(model: BertToxic, test_loader: DataLoader, criterion: torch.nn.modules.loss, device: str):
    """
    Evaluate the performance of the given model on the test dataset.

    Args:
        model (BertToxic): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.modules.loss): Loss function to use for evaluation.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the average loss and accuracy.
    """
    model.eval()
    total_correct, total_loss, total_samples = _calculate_loss(model, test_loader, criterion, device)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    log.info(f"Evaluation complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


def _calculate_loss(model: BertToxic, test_loader: DataLoader, criterion: torch.nn.modules.loss, device: str):
    """
    Calculate the total loss and accuracy for the given model on the test dataset.

    Args:
        model (BertToxic): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.modules.loss): Loss function to use for evaluation.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the total correct predictions, total loss, and total samples.
    """
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0]["input_ids"].squeeze(1).to(device)
            attention_mask = batch[0]["attention_mask"].squeeze(1).to(device)
            labels = batch[1].to(device).float()  # Ensure labels are float for BCEWithLogitsLoss

            logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).long()
            loss = criterion(logits, labels)

            total_correct += (predictions == labels.long()).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_correct, total_loss, total_samples
