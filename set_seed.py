import numpy as np
import torch
import random

def set_seed(seed_value):

    random.seed(seed_value)

    torch.manual_seed(seed_value)

    torch.cuda.manual_seed(seed_value)

    torch.cuda.manual_seed_all(seed_value)

    np.random.seed(seed_value)

