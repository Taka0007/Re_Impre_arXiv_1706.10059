# pgportfolio/learner/osbl.py
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

def get_random_loader(dataset, batch_size):
    indices = torch.randperm(len(dataset))
    return DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices))
