import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration.config import settings

model_settings = settings["model"]

class BERTEmbeddings(nn.Module):
    "BERT Embedding layer."
    def __init__(self):
        super(BERTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(model_settings["vocab_size"], model_settings["hidden_size"])
        self.position_embeddings = nn.Embedding(model_settings["max_position_embeddings"], model_settings["hidden_size"])
        self.token_type_embeddings = nn.Embedding(2, model_settings["hidden_size"])  # Segment A and B not really neccessary for single sentence classification
        self.LayerNorm = nn.LayerNorm(model_settings["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(positions) + self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BERTLayer(nn.Module):
    "Single layer of BERT consisting of self-attention and feed-forward network."
    def __init__(self):
        super(BERTLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=model_settings["hidden_size"], num_heads=model_settings["num_heads"], batch_first=True) # This is different from hf implementation, considered in weight loading
        self.intermediate = nn.Linear(model_settings["hidden_size"], model_settings["hidden_size"]*4) # Feedforward after attention
        self.output = nn.Linear(model_settings["hidden_size"]*4, model_settings["hidden_size"])   
        self.attention_layer_norm = nn.LayerNorm(model_settings["hidden_size"], eps=1e-12)     # Paper does not specifically mention this but "Attention is all you need" does
        self.output_layer_norm = nn.LayerNorm(model_settings["hidden_size"], eps=1e-12)        # same
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            key_padding_mask = (attention_mask==0).to(torch.bool)
        else:
            key_padding_mask = None
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states, key_padding_mask=key_padding_mask)
        hidden_states = self.attention_layer_norm(hidden_states + self.dropout(attention_output))  # Residual connection

        intermediate_output = F.gelu(self.intermediate(hidden_states)) # Paper specifies gelu instead of relu
        layer_output = self.output(intermediate_output)
        return self.output_layer_norm(hidden_states + self.dropout(layer_output))  # Residual connection

class Bert(nn.Module):
    """BERT model consisting of embeddings and multiple layers of BERT.
    """
    def __init__(self):
        super(Bert, self).__init__()
        self.embeddings = BERTEmbeddings()
        self.encoder = nn.ModuleList([BERTLayer() for _ in range(model_settings["num_layers"])])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        embeddings = self.embeddings(input_ids, token_type_ids)
        hidden_states = embeddings

        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states

class BertTuned(nn.Module):
    """
    BERT model extended with a classification head for toxic comment classification.
    """
    def __init__(self, bert_model, num_labels):
        super(BertTuned, self).__init__()
        self.bert = bert_model  # Your custom BERT model
        self.classifier = nn.Linear(model_settings["hidden_size"], num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Forward pass through BERT
        hidden_states = self.bert(input_ids, token_type_ids, attention_mask)
        
        # Use the last hidden state of the [CLS] token for classification
        cls_token_state = hidden_states[:, 0, :]
        cls_token_state = self.dropout(cls_token_state)
        
        # Classification head
        logits = self.classifier(cls_token_state)
        return logits
    