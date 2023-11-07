import random
import numpy as np
import torch


def fix_seed(seed, set_cuda=False, cudnn_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed(seed)
        if cudnn_deterministic:
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.deterministic = True
            # noinspection PyUnresolvedReferences
            torch.backends.cudnn.benchmark = False
