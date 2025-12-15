#  /data/downloader.py
import yfinance as yf
import pandas as pd

def download(symbols, start, end):
    df = yf.download(symbols, start=start, end=end)["Close"]
    df = df.dropna(how="any")
    return df
