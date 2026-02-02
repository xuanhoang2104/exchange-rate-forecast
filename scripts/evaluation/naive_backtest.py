import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

series = pd.read_csv(
    "data/processed/model_ready/usd_series.csv",
    index_col="date"
)["usd_index"].values

window = 500
horizon = 7
errors = []

for i in range(window, len(series)-horizon):
    last = series[i]   # naive: hôm nay = hôm trước
    true = series[i+horizon]
    errors.append(abs(true - last))

print("Naive MAE:", np.mean(errors))
