import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configuration.config import settings, get_logger
from data.finetuning.transform import bert_tokenize
from data.utils import load_data
from finetuning.evaluate import evaluate
from model.initialize import initialize_with_weights
from model.model import Bert, BertToxic

log = get_logger(__name__)


def finetune(dataset: str, dataset_dir: str, weights_dir: str = None, config: dict = settings["finetuning"]):
    """
    Fine-tunes a BERT model on a given dataset.

    Args:
        dataset (str): The name of the dataset to use for fine-tuning.
        dataset_dir (str): The directory where the dataset is located.
        weights_dir (str, optional): The directory to save the model weights. Defaults to None.
        config (dict, optional): Configuration settings for training. Defaults to settings["finetuning"].

    Returns:
        None
    """
    log.info("Loading data...")
    train_loader, test_loader = _get_data_loader(dataset, dataset_dir)

    log.info("Initializing BERT model...")
    model = _initialize_model(config)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in tqdm(range(config["num_epochs"]), desc="Fine-tuning", unit="epoch"):
        model.train()
        total_loss, total_samples = _finetune_one_epoch(config, criterion, model, optimizer, train_loader)
        avg_loss = total_loss / total_samples
        log.info(f'Epoch {epoch + 1}/{config["num_epochs"]} - Loss: {avg_loss:.4f}')

    if weights_dir:
        log.info("Saving model weights")
        torch.save(model.state_dict(), f"{weights_dir}/bert_toxic_weights.pth")

    log.info("Evaluating trained model")
    evaluate(model, test_loader, criterion, config["device"])


def _get_data_loader(dataset: str, dataset_dir: str, config: dict = settings["finetuning"]) -> tuple[
    DataLoader, DataLoader]:
    """
    Loads the training and testing data loaders.

    Args:
        dataset (str): The name of the dataset to use.
        dataset_dir (str): The directory where the dataset is located.
        config (dict, optional): Configuration settings for data loading. Defaults to settings["finetuning"].

    Returns:
        tuple[DataLoader, DataLoader]: The training and testing data loaders.
    """
    train_dataset, test_dataset = load_data(
        dataset=dataset,
        dataset_dir=dataset_dir,
        transformation=bert_tokenize,
        n_train=config["n_train"],
        n_test=config["n_test"]
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    return train_loader, test_loader


def _initialize_model(config: dict = settings["finetuning"]) -> BertToxic:
    """
    Initializes the BERT model with pre-trained weights and adapts it for binary classification.

    Args:
        config (dict, optional): Configuration settings for model initialization. Defaults to settings["finetuning"].

    Returns:
        BertToxic: The initialized BERT model for binary classification.
    """
    model = Bert()
    initialize_with_weights(model)
    model = BertToxic(model, num_labels=1)
    model.to(config["device"])
    return model


def _finetune_one_epoch(config, criterion, model, optimizer, train_loader):
    """
    Performs one epoch of fine-tuning on the training data.

    Args:
        config (dict): Configuration settings for training.
        criterion (nn.Module): The loss function.
        model (nn.Module): The BERT model to fine-tune.
        optimizer (optim.Optimizer): The optimizer for training.
        train_loader (DataLoader): The data loader for the training data.

    Returns:
        tuple[float, int]: The total loss and the total number of samples processed.
    """
    total_loss = 0
    total_samples = 0
    for batch in train_loader:
        # Tokenized inputs and labels
        input_ids = batch[0]["input_ids"].squeeze(1).to(config["device"])  # Remove extra dimension
        attention_mask = batch[0]["attention_mask"].squeeze(1).to(config["device"])
        labels = batch[1].to(config["device"]).float()  # Ensure labels are float for BCEWithLogitsLoss

        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)  # Ensure shape is (batch_size)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss, total_samples
