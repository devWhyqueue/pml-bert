import os
from typing import Tuple

import torch.distributed as dist
from torch.distributed import get_rank, is_initialized


def get_available_cpus(max_cpus: int = 32) -> int:
    """
    Get the number of available CPUs.

    This function determines the number of available CPUs by considering the
    total CPU count of the system and the SLURM_CPUS_PER_TASK environment variable.

    Args:
        max_cpus (int): The maximum number of CPUs to consider. Defaults to 32.

    Returns:
        int: The number of available CPUs minus one.
    """
    cpu_count = os.cpu_count()
    slurm_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", max_cpus))
    available = min(cpu_count, slurm_cpus)
    return max(1, available - 1)


def is_main_process() -> bool:
    """
    Check if the current process is the main process.

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return not is_initialized() or get_rank() == 0


def _initialize_distributed() -> Tuple[int, int]:
    """
    Initializes the distributed environment.

    Returns:
        Tuple[int, int]: The rank and world size.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return rank, world_size
