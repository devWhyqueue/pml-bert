import torch.nn as nn


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        # Define a simple layer for illustration; replace with actual BERT layers based on configuration
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

    def forward(self, input_ids, attention_mask=None):
        # Implement forward pass; using the dense layer as an example
        output = self.dense(input_ids)
        return output
