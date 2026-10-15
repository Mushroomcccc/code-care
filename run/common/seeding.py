import os


import random
import numpy as np

try:
    import torch
except Exception:  # NumPy-only runners do not require PyTorch.
    torch = None


def set_seed(seed: int, deterministic: bool = True):
    seed = int(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic:
        torch.use_deterministic_algorithms(True)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #可选：关闭 TF32，提高数值一致性
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
