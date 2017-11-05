## 実行手順

まず、Jupyterを立ち上げる。必要なファイルをDL&インストールするので、時間がかかる。

```bash
$ docker-compose build
$ docker-compose up
```

`historical_data` 以下にCSVファイルを置く。形式は「日付、始値、高値、安値、終値、取引量」の順。
`Keras-RL_DQN_FX.ipynb` から実行する際、上記のファイル名を引数に与えて実行すればOK。