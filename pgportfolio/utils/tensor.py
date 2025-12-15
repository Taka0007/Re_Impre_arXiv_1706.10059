# /pgportfolio/utils/tensor.py
import torch
import numpy as np

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)
