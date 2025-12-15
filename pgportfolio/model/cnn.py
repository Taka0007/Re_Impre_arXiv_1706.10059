# pgportfolio/model/cnn.py
from .base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class EIIECNN(BaseModel):
    def __init__(self, assets, window):
        super().__init__()
        self.conv1 = nn.Conv1d(window, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * assets, 128)
        self.fc2 = nn.Linear(128, assets)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)
