# /pgportfolio/model/rnn.py
from .base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class EIIRNN(BaseModel):
    def __init__(self, assets, hidden):
        super().__init__()
        self.rnn = nn.RNN(input_size=assets, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, assets)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1]
        return torch.softmax(self.fc(out), dim=1)
