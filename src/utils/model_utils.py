import os
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


def save_image(image_dir, image_name, image):
    if not os.path.exists(image_dir):
        os.makedirs(f"{image_dir}/images/")

    image.save(f"{image_dir}/images/{image_name.split('.')[0]}.png")


def save_table(table_dir, table_name, table):
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)

    table.to_csv(
        f"{table_dir}/{table_name}.csv",
        index=None,
    )


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        return result, end - start

    return wrapper
