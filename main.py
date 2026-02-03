import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="USD Index Deep Dive", layout="wide")

st.title("📈 USD Index Dashboard & Forecast")
st.write("Demo quy trình xử lý dữ liệu từ 7 Majors đến Model-Ready Data.")

# -----------------------------
# 1. Load Data
# -----------------------------
@st.cache_data
def load_all_data():
    df_clean = pd.read_csv(r"D:\FPT\ki 7\DAT\exchange-rate\exchange-rate\data\processed\merged\cleaned.csv", index_col="date", parse_dates=True)
    df_index = pd.read_csv(r"D:\FPT\ki 7\DAT\exchange-rate\exchange-rate\data\processed\usd\usd_index.csv", index_col="date", parse_dates=True)
    df_series = pd.read_csv(r"D:\FPT\ki 7\DAT\exchange-rate\exchange-rate\data\processed\model_ready\usd_series.csv", index_col="date", parse_dates=True)
    return df_clean, df_index, df_series

try:
    df_clean, df_index, df_series = load_all_data()
    model = joblib.load(r"D:\FPT\ki 7\DAT\exchange-rate\exchange-rate\models\arima_usd.pkl")
except Exception as e:
    st.error(f"Lỗi khi load dữ liệu hoặc model: {e}")
    st.stop()

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("Cấu hình Dự báo")
horizon = st.sidebar.slider("Số ngày dự đoán (Horizon)", 1, 30, 14)
history_len = st.sidebar.slider("Số ngày hiển thị lịch sử", 100, 1000, 500)

# -----------------------------
# 2. Tabs for different stages
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 7 Majors vs Index", "📈 Stationary Data", "🔮 Forecast"])

with tab1:
    st.subheader("Giai đoạn 1 & 2: Gom 7 Majors thành USD Index")
    majors = ['euro_to_usd', 'japanese_yen_to_usd', 'uk_pound_to_usd', 
              'swiss_franc_to_usd', 'australian_dollar_to_usd', 
              'canadian_dollar_to_usd', 'chinese_yuan_to_usd']
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        df_clean[majors].tail(history_len).plot(ax=ax1, alpha=0.6)
        ax1.set_title("Top 7 Major Currencies vs USD")
        st.pyplot(fig1)
    
    with col2:
        st.info("**USD Index** được tính bằng trung bình cộng (mean) của 7 đồng tiền này giúp giảm 'nhiễu' từ một đồng tiền riêng lẻ.")
        st.metric("Giá trị Index hiện tại", f"{df_index['usd_index'].iloc[-1]:.4f}")

with tab2:
    st.subheader("Giai đoạn 3: Biến đổi dữ liệu 'Dừng' (Stationary)")
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df_index["usd_index"].tail(history_len).plot(ax=ax2, color='red')
        ax2.set_title("Original USD Index (Non-Stationary)")
        st.pyplot(fig2)
        
    with col_s2:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        df_series["usd_index"].tail(history_len).plot(ax=ax3, color='green')
        ax3.set_title("Log-Diff USD Series (Stationary)")
        st.pyplot(fig3)
    
    st.success("✅ Model ARIMA/LSTM sẽ học trên dữ liệu **màu xanh** (Stationary) vì nó ổn định và dễ dự báo xu hướng thay đổi hơn.")

with tab3:
    st.subheader("Giai đoạn 4: Kết quả Dự báo từ Model")
    
    # Forecast logic (ARIMA)
    last_usd = df_index["usd_index"].iloc[-1]
    last_log = np.log(last_usd)
    
    pred_diff = model.predict(n_periods=horizon)
    pred_log = last_log + np.cumsum(pred_diff)
    pred_usd = np.exp(pred_log)
    
    future_dates = pd.date_range(start=df_index.index[-1] + pd.Timedelta(days=1), periods=horizon)
    forecast_df = pd.DataFrame({"date": future_dates, "usd_index_pred": pred_usd}).set_index("date")
    
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(df_index.index[-history_len:], df_index["usd_index"].iloc[-history_len:], label="Historical")
    ax4.plot(forecast_df.index, forecast_df["usd_index_pred"], label="Forecast", linestyle="--", color="red", marker='o')
    ax4.set_title("USD Index Forecast (ARIMA Model)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)
    
    # Summary
    change = (forecast_df.iloc[-1,0] - last_usd) / last_usd * 100
    st.write(f"### Dự báo trong {horizon} ngày tới:")
    if change > 0:
        st.success(f"📈 Xu hướng: **TĂNG** (~{change:.2f}%)")
    else:
        st.error(f"📉 Xu hướng: **GIẢM** (~{abs(change):.2f}%)")
    
    st.dataframe(forecast_df.T)

