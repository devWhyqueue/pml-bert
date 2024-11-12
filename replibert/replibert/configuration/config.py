import logging.config
from logging import Logger
from pathlib import Path

import torch.multiprocessing
import yaml
from transformers import logging as tfl

# App Config
with open(Path(__file__).parent / "app.yaml", "r") as file:
    settings = yaml.safe_load(file)

# Logging Config
logging.config.fileConfig(Path(__file__).parent / "logging.ini", disable_existing_loggers=False)
# Disable transformers logging
tfl.set_verbosity_error()
# Torch
torch.multiprocessing.set_sharing_strategy('file_system')


def get_logger(name: str) -> Logger:
    return logging.getLogger(f"replibert.{name}")
