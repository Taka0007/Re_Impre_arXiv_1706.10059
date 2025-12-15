# /pgportfolio/core/backtester.py
import torch
import numpy as np

def backtest(model, xs, ys, device, commission):
    model.eval()
    wealth = 1.0
    prev = None
    with torch.no_grad():
        for i in range(len(xs)):
            x = xs[i : i + 1].to(device)
            y = ys[i].to(device)
            w = model(x).cpu()
            ret = (w * y).sum().item()
            cost = commission * (w - (prev if prev is not None else w)).abs().sum().item()
            wealth *= (ret - cost)
            prev = w
    return wealth
