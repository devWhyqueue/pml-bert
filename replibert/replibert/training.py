import torch.optim as optim

from data_loader import CustomDataset
from model import Bert


def train(config):
    dataset = CustomDataset(config['data']['path'])
    model = Bert(config['model'])
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop skeleton
    for epoch in range(config['training']['num_epochs']):
        for data in dataset:
            # Implement forward and backward pass
            pass
