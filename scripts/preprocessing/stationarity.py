import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/usd/usd_index.csv",
                 index_col="date", parse_dates=True)

series = np.log(df["usd_index"]).diff().dropna()
series.to_csv("data/processed/model_ready/usd_series.csv")

"""
tăng từ 100 → 110
tăng từ 200 → 220

thành 
100 → 110 (tăng 10%)
200 → 220 (tăng 10%)
"""