import torch
from configuration.config import get_logger

log = get_logger(__name__)

def evaluate(model, test_loader, device, criterion):
    log.info("Starting evaluation...")
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in test_loader:
            input_ids = batch[0]["input_ids"].squeeze(1).to(device)
            attention_mask = batch[0]["attention_mask"].squeeze(1).to(device)
            labels = batch[1].to(device).float()  # Ensure labels are float for BCEWithLogitsLoss

            # Forward pass to get logits
            logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)

            # Compute loss
            loss = criterion(logits, labels)

            # Convert logits to probabilities and make predictions
            probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
            predictions = (probabilities > 0.5).long()  # Threshold at 0.5

            # Calculate number of correct predictions
            total_correct += (predictions == labels.long()).sum().item()
            total_samples += labels.size(0)

            # Track total loss
            total_loss += loss.item() * labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    log.info(f"Evaluation complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
