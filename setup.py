# Re_Impre_arXiv_1706.10059/setup.py
from setuptools import setup, find_packages

setup(
    name="pgportfolio_pytorch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "yfinance",
        "tqdm",
        "scipy",
    ],
    description="A PyTorch implementation of PGPortfolio (arXiv.1706.10059)",
)
