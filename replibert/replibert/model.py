import torch
import torch.nn as nn
import torch.nn.functional as F
from model_init import initialize_from_hf_model

class CONFIG:
    "BERT Configuration class."
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, # vocab size for hf is slightly different
                 num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

class BERTEmbeddings(nn.Module):
    "BERT Embedding layer."
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)  # Segment A and B not really neccessary for single sentence classification
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
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
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, batch_first=True) # This is different from hf implementation, considered in weight loading
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size) # Feedforward after attention
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)   
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)     # Paper does not specifically mention this but "Attention is all you need" does
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)        # same
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = self.attention_layer_norm(hidden_states + self.dropout(attention_output))  # Residual connection

        intermediate_output = F.gelu(self.intermediate(hidden_states)) # Paper specifies gelu instead of relu
        layer_output = self.output(intermediate_output)
        return self.output_layer_norm(hidden_states + self.dropout(layer_output))  # Residual connection

class BERT(nn.Module):
    "BERT model consisting of embeddings and multiple layers of BERT."
    def __init__(self, config):
        super(BERT, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = nn.ModuleList([BERTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, token_type_ids=None):
        embeddings = self.embeddings(input_ids, token_type_ids)
        hidden_states = embeddings

        for layer in self.encoder:
            hidden_states = layer(hidden_states)

        return hidden_states
    
"""Model architecture in Paper:
Attention            |
Dropout              |  All in Attention for Hugging Face, so same model
layer_norm with res  |

Linear up
gelu
Linear down
dropout
layer norm with res



"""

# Example usage
if __name__ == "__main__":
    config = CONFIG()
    model = BERT(config)
    initialize_from_hf_model(model, config)

    # Example input
    input_ids = torch.randint(0, config.vocab_size, (1, 10))  # Batch size of 1, sequence length of 10, note that first Token should be [CLS]
    token_type_ids = torch.zeros_like(input_ids)  # Segment IDs

    output = model(input_ids, token_type_ids)
    print(output.shape)  # Should be (1, 10, 768)
