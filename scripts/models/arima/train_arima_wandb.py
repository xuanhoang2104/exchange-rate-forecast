import pandas as pd
import matplotlib.pyplot as plt
import wandb
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

# 1. Init run
wandb.init(
    project="exchange-rate-2",
    name="ARIMA-baseline-usd"
)

# 2. Load data
series = pd.read_csv(
    "data/processed/model_ready/usd_series.csv",
    index_col="date"
)["usd_index"]

split = int(len(series)*0.8)
train, test = series[:split], series[split:]

# 3. Train ARIMA
model = auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=0,
    seasonal=False,
    suppress_warnings=True
)

# 4. Predict
pred = model.predict(n_periods=len(test))
mae = mean_absolute_error(test, pred)

# 5. Log metrics
wandb.log({
    "mae": mae,
    "aic": model.aic(),
    "bic": model.bic(),
    "model": "ARIMA",
    "horizon": 7,
    "input": "USD index"
})

# 6. Plot
plt.figure(figsize=(12,4))
plt.plot(test.index, test, label="True")
plt.plot(test.index, pred, label="ARIMA Forecast")
plt.legend()
plt.title("ARIMA Baseline Forecast")
plt.tight_layout()
plt.savefig("arima_forecast.png")

# 7. Upload figure
wandb.log({"forecast_plot": wandb.Image("arima_forecast.png")})

wandb.finish()
print("Logged to W&B. MAE:", mae)
