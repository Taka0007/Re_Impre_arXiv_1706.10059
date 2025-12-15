# /main.py
import json
import torch
from data.downloader import download
from data.preprocess import price_tensor
from torch.utils.data import TensorDataset
from pgportfolio.utils.tensor import to_tensor
from pgportfolio.model.cnn import EIIECNN
from pgportfolio.model.rnn import EIIRNN
from pgportfolio.model.lstm import EIILSTM
from pgportfolio.model.pvm import PortfolioVectorMemory
from pgportfolio.core.trainer import Trainer
from pgportfolio.learner.reward import immediate_reward
from pgportfolio.learner.osbl import get_random_loader
from pgportfolio.core.backtester import backtest

def load_config(path):
    return json.load(open(path))

def build_model(cfg, assets):
    if cfg["model"] == "cnn":
        return EIIECNN(assets, cfg["window"])
    if cfg["model"] == "rnn":
        return EIIRNN(assets, 64)
    if cfg["model"] == "lstm":
        return EIILSTM(assets, 64)

config = load_config("configs/default.json")

df = download(["BTC-USD","ETH-USD","XRP-USD"], start="2019-01-01", end="2025-01-01")
X, Y = price_tensor(df, config["window"])

xs = to_tensor(X)
ys = to_tensor(Y)

dataset = TensorDataset(xs, ys)
loader = get_random_loader(dataset, config["batch_size"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(config, xs.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

trainer = Trainer(model, optimizer, immediate_reward, config)

for epoch in range(config["train_epochs"]):
    avg = trainer.train(loader, device)
    print(f"Epoch {epoch+1}, avg reward={avg:.5f}")

final_wealth = backtest(model, xs, ys, device, config["commission"])
print("Final wealth:", final_wealth)
