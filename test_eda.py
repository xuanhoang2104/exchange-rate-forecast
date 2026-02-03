import pandas as pd
import matplotlib.pyplot as plt
# Load data
df_clean = pd.read_csv(r"D:\FPT\ki 7\DAT\exchange-rate\exchange-rate\data\processed\merged\cleaned.csv", index_col='date', parse_dates=True)
df_index = pd.read_csv(r"D:\FPT\ki 7\DAT\exchange-rate\exchange-rate\data\processed\usd\usd_index.csv", index_col='date', parse_dates=True)
df_series = pd.read_csv(r"D:\FPT\ki 7\DAT\exchange-rate\exchange-rate\data\processed\model_ready\usd_series.csv", index_col='date', parse_dates=True)
# Trực quan hóa
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
# 1. Các đồng tiền chính
majors = [    'euro_to_usd',
    'japanese_yen_to_usd',
    'uk_pound_to_usd',
    'swiss_franc_to_usd',
    'australian_dollar_to_usd',
    'canadian_dollar_to_usd',
    'chinese_yuan_to_usd']
df_clean[majors].plot(ax=ax[0], title="Top 7 Majors vs USD")
# 2. USD Index (Trung bình)
df_index.plot(ax=ax[1], color='red', title="USD Index (Combined Average)")
# 3. Model Ready Data (Stationary)
df_series.plot(ax=ax[2], color='green', title="USD Series (Model Ready - Stationarized)")
plt.tight_layout()
plt.show()