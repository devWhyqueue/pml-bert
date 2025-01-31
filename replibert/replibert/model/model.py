from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration.config import settings


class BertEmbeddings(nn.Module):
    """
    BERT Embedding layer.

    This layer includes word embeddings, position embeddings, and token type embeddings.
    It also applies layer normalization and dropout to the embeddings.
    """

    def __init__(self, config: Dict[str, Any] = settings["model"]) -> None:
        """
        Initialize the BERTEmbeddings layer.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters.
        """
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the BERTEmbeddings layer.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            token_type_ids (Optional[torch.Tensor]): Tensor of token type IDs. Defaults to None.

        Returns:
            torch.Tensor: Tensor of embeddings.
        """
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(
            input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = (
                self.word_embeddings(input_ids)
                + self.position_embeddings(positions)
                + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertLayer(nn.Module):
    """
    Single layer of BERT consisting of self-attention and feed-forward network.
    """

    def __init__(self, config: Dict[str, Any] = settings["model"]) -> None:
        """
        Initialize the BERTLayer.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters.
        """
        super(BertLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden_size"], num_heads=config["num_heads"], batch_first=True
        )
        self.intermediate = nn.Linear(config["hidden_size"], config["hidden_size"] * 4)
        self.output = nn.Linear(config["hidden_size"] * 4, config["hidden_size"])
        self.attention_layer_norm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.output_layer_norm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the BERTLayer.

        Args:
            hidden_states (torch.Tensor): Tensor of hidden states.
            attention_mask (Optional[torch.Tensor]): Tensor of attention mask. Defaults to None.

        Returns:
            torch.Tensor: Tensor of output hidden states.
        """
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0).to(torch.bool)
        else:
            key_padding_mask = None
        attention_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states, key_padding_mask=key_padding_mask
        )
        hidden_states = self.attention_layer_norm(hidden_states + self.dropout(attention_output))

        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        return self.output_layer_norm(hidden_states + self.dropout(layer_output))


class Bert(nn.Module):
    """
    BERT model consisting of embeddings and multiple layers of BERT.
    """

    def __init__(self, config: Dict[str, Any] = settings["model"]) -> None:
        """
        Initialize the Bert model.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters.
        """
        super(Bert, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = nn.ModuleList([BertLayer(config) for _ in range(config["num_layers"])])
        self.config = config

    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Bert model.

        Args:
            input_ids (Optional[torch.Tensor]): Tensor of input token IDs.
            token_type_ids (Optional[torch.Tensor]): Tensor of token type IDs. Defaults to None.
            attention_mask (Optional[torch.Tensor]): Tensor of attention mask. Defaults to None.
            inputs_embeds (Optional[torch.Tensor]): Precomputed embeddings. Defaults to None.

        Returns:
            torch.Tensor: Tensor of output hidden states.
        """
        # Use inputs_embeds if provided; otherwise, generate embeddings from input_ids
        if inputs_embeds is None:
            embeddings = self.embeddings(input_ids, token_type_ids)
        else:
            embeddings = inputs_embeds

        hidden_states = embeddings

        # Pass through all encoder layers
        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states



class BertToxic(nn.Module):
    """
    BERT model extended with a classification head for toxic comment classification.
    """

    def __init__(self, bert_model: Bert, num_labels: int, config: Dict[str, Any] = settings["model"]) -> None:
        """
        Initialize the BertToxic model.

        Args:
            bert_model (Bert): Pre-trained BERT model.
            num_labels (int): Number of labels for classification.
            config (Dict[str, Any]): Configuration dictionary containing model parameters.
        """
        super(BertToxic, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config["hidden_size"], num_labels)

    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the BertToxic model.

        Args:
            input_ids (Optional[torch.Tensor]): Tensor of input token IDs.
            token_type_ids (Optional[torch.Tensor]): Tensor of token type IDs. Defaults to None.
            attention_mask (Optional[torch.Tensor]): Tensor of attention mask. Defaults to None.
            inputs_embeds (Optional[torch.Tensor]): Precomputed embeddings. Defaults to None.

        Returns:
            torch.Tensor: Tensor of classification logits.
        """
        # Pass inputs_embeds to the Bert model if provided
        hidden_states = self.bert(input_ids, token_type_ids, attention_mask, inputs_embeds)
        cls_token_state = hidden_states[:, 0, :]  # [CLS] token hidden state
        cls_token_state = self.dropout(cls_token_state)
        logits = self.classifier(cls_token_state)
        return logits
