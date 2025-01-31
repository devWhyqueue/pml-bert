import torch
import torch.distributed as dist
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from captum.attr import IntegratedGradients, visualization as viz
from transformers import BertTokenizer

from tqdm import tqdm
import os

from configuration.config import get_logger, settings
from data.finetuning.datasets import FineTuningDataset
from model.model import BertToxic, Bert
from utils import is_main_process, _initialize_distributed, get_available_cpus

log = get_logger(__name__)

def explain_false_predictions(weights, dataset, save_path, config: dict = settings["finetuning"]):
    """
    Analyzes false predictions of a fine-tuned BERT model for toxic comment classification.

    This function evaluates the model using a given dataset and identifies instances where 
    the model's predicted labels differ from the true labels. For false predictions where the 
    confidence level is below 0.7, the function generates an explanation for the 
    prediction using attribution methods and logs the relevant information.

    Args:
        weights (str): Path to the saved model weights file.
        dataset (Dataset): The dataset object containing the test samples and corresponding labels.
        save_path (str): Path where the visualization file will be saved.
        config (dict, optional): Configuration dictionary specifying the device and other 
                                 finetuning settings. Defaults to `settings["finetuning"]`.

    Returns:
        None
    """
    from torch.utils.data import DataLoader

    model = BertToxic(Bert(), num_labels=1)
    state_dict = torch.load(weights, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(config["device"])
    model.eval()

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)


    records = []
    for idx, (input, label) in enumerate(test_loader):
        input_c = input.to(config["device"])
        true_label = label.to(config["device"]).item()>0.5
        prediction = torch.sigmoid(model(input_ids=input_c[:,0,:], attention_mask=input_c[:,1,:]).squeeze(-1))
        predicted_label = prediction.item()>0.5
        if true_label!=predicted_label and prediction < 0.7:
            prediction, attribution, input_ids = explain_prediction(model, input_c, tokenizer)
            score_vis = visualize_explanation(attribution,prediction, predicted_label, true_label, tokenizer, input_ids)
            if score_vis!=None:
                records.append(score_vis)
    
            print("True Label: ", label.item(), "Prediction: ", prediction, "Explanation: ", attribution)

    visualization = viz.visualize_text(records)
    # Save as an HTML file
    with open(save_path + "explain.html", "w") as f:
        f.write(visualization.data)

def explain_prediction(weights, input, tokenizer, config: dict = settings["finetuning"]):
    """
    Generates predictions and explanations for a given input using a BERT-based model.

    Args:
        weights (Union[BertToxic, str]): The model instance or a path to the saved model weights.
        input (Union[str, torch.Tensor]): The input to classify and explain. If a string, it will be tokenized. 
                                          If a tensor, it should already contain token IDs and attention masks.
        tokenizer (BertTokenizer): The tokenizer used for tokenizing input texts.
        config (dict): Configuration settings including the device ("cuda" or "cpu").

    Returns:
        Tuple[float, dict, torch.Tensor]:
            - Probability of the positive class (toxic).
            - Token-level attributions as a dictionary where keys are tokens and values are attribution scores.
            - Input IDs used for explanation visualization.
    """

    if isinstance(weights, BertToxic):
        model = weights.to(config["device"])
    else:
        model = BertToxic(Bert().to(config["device"]), num_labels=1)
        state_dict = torch.load(weights, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(config["device"])


    if isinstance(input, str):
        tokenized = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=settings["model"]["max_position_embeddings"])
        input_ids = tokenized["input_ids"].to(config["device"])
        attention_mask = tokenized["attention_mask"].to(config["device"])
    elif isinstance(input, torch.Tensor):
        input_ids = input[:, 0, :]
        attention_mask = input[:, 1, :]
    else:
        raise ValueError("Input must be a string or a preprocessed tensor.")

    def forward_with_embeddings(embedding_inputs, attention_mask):
        output = model(input_ids=None, attention_mask=attention_mask, inputs_embeds=embedding_inputs)
        
        return output

    ig = IntegratedGradients(forward_with_embeddings)

    with torch.no_grad():
        embedding_inputs = model.bert.embeddings(input_ids=input_ids)
        logit = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
        probability = torch.sigmoid(logit).item()
        baseline_embeds = create_baseline(input_ids, tokenizer, model)

    attributions, _ = ig.attribute(
        inputs=embedding_inputs.requires_grad_(),
        additional_forward_args=(attention_mask,),
        target=None,
        baselines=baseline_embeds,
        return_convergence_delta=True
    )

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    token_attributions = {
        token: attr.item() for token, attr in zip(tokens, attributions[0].sum(dim=-1).squeeze())
    }


    return probability, token_attributions, input_ids

def visualize_explanation(token_attributions, pred_prob, pred_class, true_label, tokenizer, input_ids):
    """
    Saves the explanation of the input using Captum's text visualization.

    Args:
        token_attributions (dict): Dictionary with tokens as keys and attribution scores as values.
        pred_prob (float): Predicted toxicity probability
        pred_class (int): predicted class
        true_label (int): true class
        tokenizer (BertTokenizer): tokenizer to decode tokens
        input_ids (Tensor): Input IDs of sample

    Returns:
        None
    """
    tokens = list(token_attributions.keys())
    attributions_sum = list(token_attributions.values())  # Token attributions
    delta = sum(attributions_sum)  # Convergence score (optional)
    full_text = tokenizer.decode(input_ids.squeeze(0).tolist(), skip_special_tokens=True)

    score_vis = viz.VisualizationDataRecord(
        word_attributions=attributions_sum,
        pred_prob=pred_prob,
        pred_class=pred_class,
        true_class=true_label, 
        attr_class=full_text, 
        attr_score=delta,       
        raw_input_ids=tokens,
        convergence_score=delta
    )

    return score_vis



def create_baseline(input_ids: torch.Tensor, tokenizer: BertTokenizer, model: BertToxic, config: dict = settings["finetuning"]):
    """
    Creates a baseline tensor using padding tokens for attribution methods while keeping special tokens unchanged.

    Args:
        input_ids (torch.Tensor): Tensor containing tokenized input IDs.
        tokenizer (BertTokenizer): Tokenizer used for tokenization.
        config (dict): Configuration dictionary containing model settings.
        model (BertToxic): Bert model

    Returns:
        torch.Tensor: Baseline tensor with the same shape as input_ids, where special tokens remain unchanged,
                      and other tokens are replaced with neutral token IDs.
    """
    pad_token_id = tokenizer.pad_token_id  # Get the padding token ID
    special_token_mask = torch.isin(input_ids, torch.tensor(tokenizer.all_special_ids, device=input_ids.device))
    
    # Create a baseline where non-special tokens are replaced with pad_token_id
    baseline_ids = input_ids.clone()
    baseline_ids[~special_token_mask] = pad_token_id
    
    with torch.no_grad():
        baseline_embeds = model.bert.embeddings(input_ids=baseline_ids.to(config["device"]))
    
    return baseline_embeds




def evaluate_submission(submissions_dir: str, dataset_split: str):
    """
    Evaluate given submission files on the Civil Comments Dataset.

    Args:
        dataset (FineTuningDataset): The dataset to evaluate on.
        submission (str): Path to the submission files.

    Returns:
        None
    """
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score

    dataset = load_dataset("civil_comments")

    true_labels = [1 if sample["toxicity"] >= 0.5 else 0 for sample in dataset[dataset_split]]
    submission_files = [os.path.join(submissions_dir, f) for f in os.listdir(submissions_dir) if f.endswith('.csv')]

    for file_path in submission_files:
        evaluate_file(true_labels, file_path)
        

def evaluate_file(true_labels, file_path: str):
    """
    Evaluate the given submission file.

    Args:
        true_labels (list): Correct labels
        file_path (str): Path to the submission file.

    Returns:
        None
    """
    def load_predictions(csv_file):
        predictions = []
        with open(csv_file, "r") as file:
            next(file)  # Skip header
            for line in file:
                _, prediction = line.strip().split(",")  # Split by comma
                predictions.append(float(prediction))   # Convert prediction to float
        return predictions
    
    predicted_values = load_predictions(file_path)

    assert len(true_labels) == len(predicted_values), f"Mismatch in true labels and predictions for {file_path}"

    roc_auc = roc_auc_score(true_labels, predicted_values, average="weighted")
    log.info(f"File: {os.path.basename(file_path)}: ")

    predicted_labels = [1 if label >=0.5 else 0 for label in predicted_values]

    class_report = classification_report(true_labels, predicted_labels)
    log.info(f"Classification Report:\n{class_report}")

    log.info(f"ROC-AUC Score: {roc_auc:.4f}")
        


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
        = _calculate_local_results(model, test_loader, config["device"])
    total_loss, total_samples = _aggregate_scalar_values(local_loss, local_samples, config["device"])
    combined_labels, combined_preds, combined_probs = _gather_all_predictions(local_labels, local_preds, local_probs)

    if is_main_process():
        _log_metrics(total_loss, total_samples, combined_labels, combined_preds, combined_probs)


def _calculate_local_results(model: torch.nn.Module, test_loader: DataLoader, device: str):
    """
    Calculate local loss and predictions for the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
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
            predictions = (probabilities > 0.5).long()
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
