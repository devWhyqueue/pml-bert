import os

from torch.distributed import get_rank, is_initialized


def get_available_cpus(max_cpus: int = 32) -> int:
    cpu_count = os.cpu_count()
    slurm_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", max_cpus))
    available = min(cpu_count, slurm_cpus)
    return available - 1


def is_main_process() -> bool:
    """
    Check if the current process is the main process.

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return not is_initialized() or get_rank() == 0
