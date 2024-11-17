from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
import yaml
from configuration.config import settings, get_logger
from data.training.dataset import TrainingDataset
from model import Bert, BertTuned
from model_init import initialize_from_hf_model
from transformers import BertTokenizer
from data.utils import load_data
from torch.utils.data import DataLoader

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


def train(config=settings): 
    log.warning("Running fine-tuning")

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Transformation function for tokenization
    def tokenize_fn(sentence):
        return tokenizer(
            sentence,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    # Load datasets
    train_dataset, test_dataset = load_data(
        dataset="civil_comments", 
        dataset_dir="/home/space/datasets/civil_comments", 
        transformation=tokenize_fn, 
        n_train=config["training"]["n_train"], 
        n_test=config["training"]["n_test"]
    )
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=16)

    # Load the pretrained BERT model with a classification head
    model = Bert(config["model"])
    initialize_from_hf_model(model, config["model"])
    model = BertTuned(model, num_labels=1, config=config["model"])

    model.to(config["training"]["device"])

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=0)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    log.warning("Training")

    for epoch in range(config["training"]["num_epochs"]):
        model.train()  # Set model to training mode
        total_loss = 0
        total_samples = 0

        for batch in train_loader:
            # Tokenized inputs and labels
            input_ids = batch[0]["input_ids"].squeeze(1).to(config["training"]["device"])  # Remove extra dimension
            attention_mask = batch[0]["attention_mask"].squeeze(1).to(config["training"]["device"])
            labels = batch[1].to(config["training"]["device"]).float()  # Ensure labels are float for BCEWithLogitsLoss

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)  # Ensure shape is (batch_size)
            loss = criterion(logits, labels)  # Compute loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        # Log epoch results
        avg_loss = total_loss / total_samples
        log.info(f'Epoch {epoch + 1}/{config["training"]["num_epochs"]} - Loss: {avg_loss:.4f}')

        # Evaluate after each epoch
        evaluate(model, test_loader, config["training"]["device"], criterion)

    log.info("Training complete!")

if __name__ == "__main__":
    with open(Path(__file__).parent / "configuration/app.yaml", "r") as file:
        settings = yaml.safe_load(file)
    train(settings)