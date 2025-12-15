# /pgportfolio/core/trainer.py
import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, loss_fn, config):
        self.model = model
        self.opt = optimizer
        self.loss_fn = loss_fn
        self.config = config

    def train(self, loader, device):
        self.model.train()
        total = 0
        prev = torch.ones(loader.dataset[0][1].size(0), device=device)[None, :] / loader.dataset[0][1].size(0)
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)

            w = self.model(x)
            r = self.loss_fn(w, y, prev, self.config["commission"])
            loss = -r.mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            prev = w.detach()
            total += r.mean().item()

        return total / len(loader)
