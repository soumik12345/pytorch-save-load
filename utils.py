import os
import torch
import shutil
import random
import numpy as np
import pyshorteners


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shorten_url(long_url):
    type_tiny = pyshorteners.Shortener()
    short_url = type_tiny.tinyurl.short(long_url)
    return short_url


def cleanup():
    shutil.rmtree("artifacts")
    os.remove("checkpoint.pth")
