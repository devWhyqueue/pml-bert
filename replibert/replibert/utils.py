import os

from torch.distributed import get_world_size, get_rank, is_initialized


def get_available_cpus(max_cpus: int = 32) -> int:
    """
    Calculate the number of available CPUs for the current process.

    This function determines the number of CPUs available for the current
    process, taking into account the world size in a distributed setting,
    the total number of CPUs on the system, and the SLURM environment
    variable `SLURM_CPUS_PER_TASK`.

    Args:
        max_cpus (int): The maximum number of CPUs to consider. Defaults to 32.

    Returns:
        int: The number of available CPUs.
    """
    world_size = abs(get_world_size())
    cpu_count = os.cpu_count()
    slurm_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", max_cpus))
    available = min(cpu_count, slurm_cpus) // world_size
    if world_size == 1 or get_rank() == 0:
        available -= 1
    return available


def is_main_process() -> bool:
    """
    Check if the current process is the main process.

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return not is_initialized() or get_rank() == 0
