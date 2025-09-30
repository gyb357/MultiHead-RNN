import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
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
        except Exception:
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_seed_list(runs: int, master: int) -> list[int]:
    rng = np.random.default_rng(master)
    seeds = rng.choice(np.arange(1, 2**31 - 1, dtype=np.int64), size=runs, replace=False)
    return seeds.tolist()


def derive_seed(base_seed: int, window: int) -> int:
    return int((base_seed * 10_000_019 + window) % (2**31 - 1))

