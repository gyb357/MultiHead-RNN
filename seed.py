import os
import random
import numpy as np
import torch
from typing import List


def make_seed_list(
        runs: int,
        master: int
) -> List[int]:
    rng = np.random.default_rng(master)
    seeds = rng.integers(1, 2**31 - 1, size=runs, dtype=np.int64)

    return seeds.tolist()


def derive_seed(
        base_seed: int,
        window: int
) -> int:
    rng = np.random.default_rng(base_seed + window)

    return int(rng.integers(1, 2**31 - 1))


def set_seed(
        seed: int,
        deterministic: bool = False
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"Warning: Could not enable deterministic algorithms: {e}")
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

