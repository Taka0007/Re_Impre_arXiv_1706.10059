# /data/preprocess.py
import numpy as np

def normalize_price(price):
    return price / price[0] - 1.0

def price_tensor(df, window):
    price = df.values
    T, N = price.shape
    X = []
    Y = []
    for t in range(window, T - 1):
        x = normalize_price(price[t - window:t])
        y = price[t + 1] / price[t]
        X.append(x)
        Y.append(y)
    return np.stack(X), np.stack(Y)
