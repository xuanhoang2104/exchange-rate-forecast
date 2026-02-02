import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="USD Index Forecast", layout="wide")

st.title("📈 USD Index Forecast (ARIMA Baseline)")
st.write("Dự đoán dựa trên ARIMA(0,0,2) + log-diff inverse transform")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(
    "data/processed/usd/usd_index.csv",
    index_col="date",
    parse_dates=True
)

# Load model
model = joblib.load("models/arima_usd.pkl")

# -----------------------------
# Sidebar
# -----------------------------
horizon = st.sidebar.slider(
    "Số ngày dự đoán",
    min_value=1,
    max_value=30,
    value=14
)

# -----------------------------
# Forecast logic
# -----------------------------
last_usd = df["usd_index"].iloc[-1]
last_log = np.log(last_usd)

pred_diff = model.predict(n_periods=horizon)

pred_log = last_log + np.cumsum(pred_diff)
pred_usd = np.exp(pred_log)

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=horizon
)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "usd_index_pred": pred_usd
}).set_index("date")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12,5))

ax.plot(df.index[-500:], df["usd_index"].iloc[-500:], label="Historical")
ax.plot(forecast_df.index, forecast_df["usd_index_pred"],
        label="Forecast", linestyle="--", color="red")

ax.set_title("USD Index Forecast")
ax.set_ylabel("USD Strength")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Table
# -----------------------------
st.subheader("📊 Forecast Table")
st.dataframe(forecast_df)

# -----------------------------
# Interpretation
# -----------------------------
change = (forecast_df.iloc[-1,0] - last_usd) / last_usd * 100

st.subheader("📌 Interpretation")

if change > 0:
    st.success(f"USD dự kiến tăng khoảng {change:.2f}% trong {horizon} ngày.")
else:
    st.error(f"USD dự kiến giảm khoảng {abs(change):.2f}% trong {horizon} ngày.")
