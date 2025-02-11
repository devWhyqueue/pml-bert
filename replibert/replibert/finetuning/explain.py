from typing import Union, Dict, Any, Tuple, List, Optional

import torch
from captum.attr import IntegratedGradients
from captum.attr._utils import visualization as viz
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from configuration.config import settings, get_logger
from model.model import BertToxic, Bert

log = get_logger(__name__)


def explain_false_predictions(weights: str, dataset: torch.utils.data.Dataset, explanation_path: str,
                              config: Dict[str, Any] = settings["finetuning"]) -> None:
    """
    Generate explanations for false predictions and save them as an HTML file.

    Args:
        weights (str): Path to the model weights.
        dataset (torch.utils.data.Dataset): Dataset to evaluate.
        explanation_path (str): Path to save the HTML file with explanations.
        config (Dict[str, Any], optional): Configuration dictionary. Defaults to settings["finetuning"].
    """
    model = _load_model(weights, config["device"])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    records = _collect_explanations(model, test_loader, tokenizer, config)
    visualization = viz.visualize_text(records)

    with open(f"{explanation_path}/explain.html", "w") as f:
        f.write(visualization.data)


def _load_model(weights: str, device: str) -> BertToxic:
    """Load a fine-tuned BertToxic model from given weights."""
    model = BertToxic(Bert(), num_labels=1)
    state_dict = torch.load(weights, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _collect_explanations(model: BertToxic, test_loader: DataLoader, tokenizer: BertTokenizer, config: Dict[str, Any]) \
        -> List[viz.VisualizationDataRecord]:
    """Collect explanations for misclassified samples with confidence above 0.7."""
    records = []
    for inputs, label in test_loader:
        inputs = inputs.to(config["device"])
        true_label = (label.to(config["device"]).item() > 0.5)
        logits = model(input_ids=inputs[:, 0, :], attention_mask=inputs[:, 1, :]).squeeze(-1)
        prob = torch.sigmoid(logits).item()
        predicted_label = prob > 0.5

        if true_label != predicted_label and prob > 0.7:
            pred_prob, attribution, input_ids = _explain_prediction(model, inputs, tokenizer, config)
            vis_record = _visualize_explanation(attribution, pred_prob, predicted_label, true_label, tokenizer,
                                                input_ids)
            if vis_record is not None:
                records.append(vis_record)

            log.info("True Label:", label.item(), "Prediction:", pred_prob, "Explanation:", attribution)

    return records


def _explain_prediction(model_or_weights: Union[BertToxic, str], data: Union[str, torch.Tensor],
                        tokenizer: BertTokenizer, config: Dict[str, Any] = settings["finetuning"]) \
        -> Tuple[float, Dict[str, float], torch.Tensor]:
    """Generate prediction probability and token-level attributions for an input."""
    model = _get_model_instance(model_or_weights, config)
    input_ids, attention_mask = _prepare_input_tensors(data, tokenizer, config)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(-1)
        probability = torch.sigmoid(logits).item()

    embedding_inputs = model.bert.embeddings(input_ids=input_ids)
    baseline_embeds = _create_baseline(input_ids, tokenizer, model, config)
    attributions = _compute_attributions(model, embedding_inputs, attention_mask, baseline_embeds)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    token_attributions = {
        token: float(attr.item())
        for token, attr in zip(tokens, attributions[0].sum(dim=-1).squeeze())
    }
    return probability, token_attributions, input_ids


def _get_model_instance(model_or_path: Union[BertToxic, str], config: Dict[str, Any]) -> BertToxic:
    """Return a BertToxic model or load from checkpoint path."""
    if isinstance(model_or_path, BertToxic):
        return model_or_path.to(config["device"])

    model = BertToxic(Bert().to(config["device"]), num_labels=1)
    state_dict = torch.load(model_or_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _prepare_input_tensors(data: Union[str, torch.Tensor], tokenizer: BertTokenizer, config: Dict[str, Any]) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert data into input_ids and attention_mask."""
    if isinstance(data, str):
        tokenized = tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings["model"]["max_position_embeddings"]
        )
        input_ids = tokenized["input_ids"].to(config["device"])
        attention_mask = tokenized["attention_mask"].to(config["device"])
    elif isinstance(data, torch.Tensor):
        input_ids = data[:, 0, :]
        attention_mask = data[:, 1, :]
    else:
        raise ValueError("Input must be a string or a preprocessed tensor.")

    return input_ids, attention_mask


def _compute_attributions(model: BertToxic, embedding_inputs: torch.Tensor, attention_mask: torch.Tensor,
                          baseline_embeds: torch.Tensor) -> torch.Tensor:
    """Compute Integrated Gradients attributions."""

    def forward_with_embeddings(embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return model(input_ids=None, attention_mask=mask, inputs_embeds=embeds)

    ig = IntegratedGradients(forward_with_embeddings)
    attributions, _ = ig.attribute(
        inputs=embedding_inputs.requires_grad_(),
        additional_forward_args=(attention_mask,),
        baselines=baseline_embeds,
        return_convergence_delta=True
    )
    return attributions


def _visualize_explanation(token_attributions: Dict[str, float], pred_prob: float, pred_class: bool,
                           true_label: bool, tokenizer: BertTokenizer, input_ids: torch.Tensor) \
        -> Optional[viz.VisualizationDataRecord]:
    """Build a Captum VisualizationDataRecord for HTML rendering."""
    tokens = list(token_attributions.keys())
    attributions_sum = list(token_attributions.values())
    delta = sum(attributions_sum)
    full_text = tokenizer.decode(input_ids.squeeze(0).tolist(), skip_special_tokens=True)

    return viz.VisualizationDataRecord(
        word_attributions=attributions_sum,
        pred_prob=pred_prob,
        pred_class=pred_class,
        true_class=true_label,
        attr_class=full_text,
        attr_score=delta,
        raw_input_ids=tokens,
        convergence_score=delta
    )


def _create_baseline(input_ids: torch.Tensor, tokenizer: BertTokenizer, model: BertToxic,
                     config: Dict[str, Any] = settings["finetuning"]) -> torch.Tensor:
    """Create baseline embeddings using pad tokens (ignoring special tokens)."""
    pad_token_id = tokenizer.pad_token_id
    special_tokens = torch.tensor(tokenizer.all_special_ids, device=input_ids.device)
    special_mask = torch.isin(input_ids, special_tokens)

    baseline_ids = input_ids.clone()
    baseline_ids[~special_mask] = pad_token_id

    with torch.no_grad():
        baseline_embeds = model.bert.embeddings(baseline_ids.to(config["device"]))

    return baseline_embeds
