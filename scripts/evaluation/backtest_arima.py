import pandas as pd
import numpy as np
import wandb
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm   # <<< thêm dòng này

wandb.init(
    project="exchange-rate-2",
    name="ARIMA-rolling-backtest"
)

series = pd.read_csv(
    "data/processed/model_ready/usd_series.csv",
    index_col="date"
)["usd_index"].values

window = 500
horizon = 7
errors = []

# total số step để tqdm tính %
total_steps = len(series) - horizon - window

for i in tqdm(range(window, len(series)-horizon), total=total_steps):
    train = series[i-window:i]
    true = series[i+horizon]

    model = auto_arima(
        train,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        d=0,
        seasonal=False,
        suppress_warnings=True
    )

    pred = model.predict(n_periods=horizon)[-1]
    error = abs(true - pred)
    errors.append(error)

mae = np.mean(errors)
std = np.std(errors)

wandb.log({
    "rolling_mae": mae,
    "rolling_std": std,
    "model": "ARIMA",
    "window": window,
    "horizon": horizon
})

wandb.finish()
print("Rolling ARIMA MAE:", mae)
