import torch
import torch.optim as optim
import torch.nn as nn
from configuration.config import settings, get_logger
from model import Bert, BertTuned
from model_init import initialize_from_hf_model
from transformers import BertTokenizer
from data.utils import load_data
from torch.utils.data import DataLoader
from evaluating import evaluate

train_settings = settings["training"]

log = get_logger(__name__)

def get_loader():
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
        n_train=train_settings["n_train"], 
        n_test=train_settings["n_test"]
    )
    train_loader = DataLoader(train_dataset, batch_size=train_settings["batch_size"], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=train_settings["batch_size"], shuffle=False, num_workers=2)

    return train_loader, test_loader


def tune(): 

    log.info("Loading data")
    train_loader, test_loader = get_loader()

    log.info("Creating model and loading pre-trained weights")
    model = Bert()
    initialize_from_hf_model(model)
    model = BertTuned(model, num_labels=1)
    model.to(train_settings["device"])

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=train_settings["learning_rate"], weight_decay=0)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    log.info("Fine-tuning")

    for epoch in range(train_settings["num_epochs"]):
        model.train()  # Set model to training mode
        total_loss = 0
        total_samples = 0

        for batch in train_loader:
            # Tokenized inputs and labels
            input_ids = batch[0]["input_ids"].squeeze(1).to(train_settings["device"])  # Remove extra dimension
            attention_mask = batch[0]["attention_mask"].squeeze(1).to(train_settings["device"])
            labels = batch[1].to(train_settings["device"]).float()  # Ensure labels are float for BCEWithLogitsLoss

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
        log.info(f'Epoch {epoch + 1}/{train_settings["num_epochs"]} - Loss: {avg_loss:.4f}')

    if train_settings["save_model"]:
        log.info("Save trained model")
        torch.save(model.state_dict(), "model_weights.pth")

    log.info("Evaluating trained model")
    evaluate(model, test_loader, train_settings["device"], criterion)

    log.info("Fine-tuning complete!")
