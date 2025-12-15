# /pgportfolio/learner/reward.py
import torch

def immediate_reward(w, price_rel, prev_w, commission):
    gross = (w * price_rel).sum(dim=1)
    cost = commission * (w - prev_w).abs().sum(dim=1)
    return torch.log(gross - cost + 1e-8)
