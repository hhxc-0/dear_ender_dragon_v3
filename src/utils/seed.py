# Set python/numpy/torch/env seeds + deterministic toggles


def seed_all(seed: int = 42, deterministic: bool = False):
    import os

    # torch imported earlier in train.py, might not work
    if deterministic:
        # For CUDA matmul determinism (often needed on recent CUDA versions)
        # Must be set BEFORE the relevant CUDA kernels are used; safest is before importing torch.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Alternative allowed value on some setups: ":16:8"

    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Force deterministic algorithms where possible
        torch.use_deterministic_algorithms(True)

        # cuDNN settings (important for convnets / RNNs)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
