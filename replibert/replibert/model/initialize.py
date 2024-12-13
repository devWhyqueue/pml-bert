from transformers import BertModel
from transformers.models.bert import BertLayer

from model.model import Bert


def initialize_with_weights(model: Bert) -> None:
    """
    Initialize a custom model with weights from a pre-trained Hugging Face BERT model.
    """
    hf_model = BertModel.from_pretrained("bert-base-uncased")

    # Copy embedding weights and biases
    state_dict = hf_model.embeddings.state_dict()
    truncated_pos_embeddings = state_dict['position_embeddings.weight'][:model.config['max_position_embeddings'], :]
    state_dict['position_embeddings.weight'] = truncated_pos_embeddings
    model.embeddings.load_state_dict(state_dict)

    # Copy encoder layers
    for layer, hf_layer in zip(model.encoder, hf_model.encoder.layer):
        _copy_attention_weights(hf_layer, layer, model.config["hidden_size"])
        _copy_ffn_weights(hf_layer, layer)
        _copy_layer_norms(hf_layer, layer)


def _copy_attention_weights(source_layer: BertLayer, target_layer: BertLayer, hidden_size: int) -> None:
    """
    Copy the attention weights and biases from the source layer to the target layer.
    """
    for i, attr in enumerate(['query', 'key', 'value']):
        weight = getattr(source_layer.attention.self, attr).weight.data
        bias = getattr(source_layer.attention.self, attr).bias.data
        start, end = i * hidden_size, (i + 1) * hidden_size
        target_layer.attention.in_proj_weight.data[start:end] = weight
        target_layer.attention.in_proj_bias.data[start:end] = bias

    target_layer.attention.out_proj.weight.data = source_layer.attention.output.dense.weight.data
    target_layer.attention.out_proj.bias.data = source_layer.attention.output.dense.bias.data


def _copy_ffn_weights(source_layer: BertLayer, target_layer: BertLayer) -> None:
    """
    Copy feed-forward network weights and biases from the source layer to the target layer.
    """
    target_layer.intermediate.weight.data = source_layer.intermediate.dense.weight.data
    target_layer.intermediate.bias.data = source_layer.intermediate.dense.bias.data
    target_layer.output.weight.data = source_layer.output.dense.weight.data
    target_layer.output.bias.data = source_layer.output.dense.bias.data


def _copy_layer_norms(source_layer: BertLayer, target_layer: BertLayer) -> None:
    """
    Copy LayerNorm parameters from the source layer to the target layer.
    """
    target_layer.attention_layer_norm.load_state_dict(source_layer.attention.output.LayerNorm.state_dict())
    target_layer.output_layer_norm.load_state_dict(source_layer.output.LayerNorm.state_dict())
