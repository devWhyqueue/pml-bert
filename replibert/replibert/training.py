import torch.optim as optim

from configuration.config import settings, get_logger
from data.training.dataset import TrainingDataset
from model import Bert

log = get_logger(__name__)


def train(config=settings):
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
