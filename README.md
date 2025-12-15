## Re_Impre_arXiv_1706.10059
https://arxiv.org/abs/1706.10059 の再現実装用のレポジトリ

## 公開されている実装
https://github.com/ZhengyaoJiang/PGPortfolio


### フォルダ構成
```
Re_Impre_arXiv_1706.10059/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── configs/
│   ├── cnn.json
│   ├── rnn.json
│   ├── lstm.json
│   ├── pvm_osbl.json
│   └── default.json
├── data/
│   ├── downloader.py
│   ├── preprocess.py
│   └── datasets.py
├── pgportfolio/
│   ├── __init__.py
│   ├── core/
│   │   ├── engine.py
│   │   ├── trainer.py
│   │   ├── backtester.py
│   │   └── utils.py
│   ├── model/
│   │   ├── base.py
│   │   ├── cnn.py
│   │   ├── rnn.py
│   │   ├── lstm.py
│   │   └── pvm.py
│   ├── learner/
│   │   ├── osbl.py
│   │   ├── optimizer.py
│   │   └── reward.py
│   └── env/
│       ├── portfolio_env.py
│       └── market_env.py
├── notebooks/
│   ├── train.ipynb
│   ├── backtest.ipynb
│   └── analysis.ipynb
└── main.py
```
