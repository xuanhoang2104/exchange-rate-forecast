import pandas as pd
import joblib
from pmdarima import auto_arima

series = pd.read_csv(
    "data/processed/model_ready/usd_series.csv",
    index_col="date"
)["usd_index"]

model = auto_arima(
    series,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=0,
    seasonal=False,
    trace=True,
    suppress_warnings=True
)

joblib.dump(model, "models/arima_usd.pkl")
print("Saved ARIMA model to models/arima_usd.pkl")
