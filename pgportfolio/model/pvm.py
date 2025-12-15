# pgportfolio/model/pvm.py
import torch
import torch.nn as nn

class PortfolioVectorMemory(nn.Module):
    def __init__(self, assets, memory_size):
        super().__init__()
        self.register_buffer("memory", torch.zeros(memory_size, assets))

    def forward(self, new):
        self.memory[:-1] = self.memory[1:].clone()
        self.memory[-1] = new.detach()
        return self.memory
