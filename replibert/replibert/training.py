import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from configuration.config import settings, get_logger
from data.training.dataset import TrainingDataset
from model import Bert, BertTuned
from model_init import initialize_from_hf_model
log = get_logger(__name__)


def train_base(config=settings):
    log.warning("Running training")

    dataset = TrainingDataset(config['data']['path'])
    model = Bert(config['model'])
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop skeleton
    for epoch in range(config['training']['num_epochs']):
        for data in dataset:
            # Implement forward and backward pass
            pass

    log.info("Completed training.")

def train(config=settings): 
    log.warning("Running training fine tuning")

    # Dataset and DataLoader
    dataset = TrainingDataset(config['data']['path'])  
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Model and optimizer
    model = Bert(config['model'])  
    initialize_from_hf_model(Bert, config['model'])
    model = BertTuned(model, num_labels= 2).to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()  

    # Training loop
    for epoch in range(config['training']['num_epochs']):
        model.train()  # Set model to training mode
        total_loss = 0
        total_samples = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")

            special_tokens_mask = batch['special_tokens_mask'].to("cuda")  
            mlm_labels = batch['mlm_labels'].to("cuda")

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs  # Assuming your model returns logits
            loss = criterion(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))  # MLM-specific loss

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
        log.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']} - Loss: {avg_loss:.4f}")

    log.info("Completed training.")