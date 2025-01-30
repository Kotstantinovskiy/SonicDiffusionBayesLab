import random
import time
from functools import wraps

import torch
from torchvision import transforms


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def setup_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)


def to_pil_image(tensor):
    return transforms.ToPILImage()(tensor)


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        return result, end - start

    return wrapper
