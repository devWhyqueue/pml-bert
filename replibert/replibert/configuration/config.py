import logging.config
from logging import Logger
from pathlib import Path

import torch.multiprocessing
import yaml
from transformers import logging as tfl

from utils import is_main_process

# App Config
with open(Path(__file__).parent / "app.yaml", "r") as file:
    settings = yaml.safe_load(file)

# Logging Config
logging.config.fileConfig(Path(__file__).parent / "logging.ini", disable_existing_loggers=False)
# Disable transformers logging
tfl.set_verbosity_error()
# Torch
torch.multiprocessing.set_sharing_strategy('file_system')


class RankFilter(logging.Filter):
    def filter(self, record):
        return is_main_process()


def get_logger(name: str) -> Logger:
    logger = logging.getLogger(f"replibert.{name}")
    logger.addFilter(RankFilter())
    return logger
