import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from configuration.config import get_logger

log = get_logger(__name__)


def evaluate(model: torch.nn.Module, test_loader: DataLoader, criterion: torch.nn.modules.loss, device: str):
    """
    Evaluate the performance of the given model on the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.modules.loss): Loss function to use for evaluation.
        device (str): Device to run the evaluation on.
    """
    model.eval()
    total_loss, total_samples, all_labels, all_predictions, all_probabilities = _calculate_loss(
        model, test_loader, criterion, device
    )

    avg_loss = total_loss / total_samples
    log.info(f"Evaluation complete - Loss: {avg_loss:.4f}")

    # Compute classification report
    class_report = classification_report(all_labels, all_predictions, target_names=["class_0", "class_1"])
    log.info(f"Classification Report:\n{class_report}")

    # Compute ROC-AUC score
    roc_auc = roc_auc_score(all_labels, all_probabilities, average="weighted")
    log.info(f"ROC-AUC Score: {roc_auc:.4f}")


def _calculate_loss(model: torch.nn.Module, test_loader: DataLoader, criterion: torch.nn.modules.loss, device: str):
    """
    Calculate the total loss and accuracy for the given model on the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.modules.loss): Loss function to use for evaluation.
        device (str): Device to run the evaluation on.

    Returns:
        tuple: A tuple containing evaluation metrics and predictions/labels.
    """
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            input_ids = inputs[:, 0, :].to(device, non_blocking=True)
            attention_mask = inputs[:, 1, :].to(device, non_blocking=True)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).long()
            loss = criterion(logits, labels)

            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            # Store predictions and labels for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_probabilities.extend(probabilities.cpu().numpy())

    return total_loss, total_samples, all_labels, all_predictions, all_probabilities
