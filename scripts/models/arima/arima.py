import pandas as pd
import numpy as np
import wandb
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

wandb.init(
    project="exchange-rate-2",
    name="ARIMA-backtest"
)

series = pd.read_csv(
    "data/processed/model_ready/usd_series.csv",
    index_col="date"
)["usd_index"]

window = 1000
horizon = 7

errors = []

for i in range(window, len(series)-horizon):
    train = series[i-window:i]
    test = series[i+horizon]

    model = auto_arima(
        train,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        d=0,
        seasonal=False,
        suppress_warnings=True
    )

    pred = model.predict(n_periods=horizon)[-1]
    error = abs(test - pred)
    errors.append(error)

mean_error = np.mean(errors)
std_error = np.std(errors)

wandb.log({
    "backtest_mae": mean_error,
    "backtest_std": std_error,
    "window": window,
    "horizon": horizon,
    "n_tests": len(errors)
})

wandb.finish()

print("Backtest MAE:", mean_error)
print("Backtest STD:", std_error)
