import pandas as pd
import numpy as np
import os

# 1. Config
input_path = "data/processed/usd/usd_index.csv"
output_path = "data/processed/model_ready/usd_series.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df = pd.read_csv(input_path, index_col="date", parse_dates=True)

# 2. Stationary Transformation (Log + Diff)
print(" Transforming USD Index to stationary series (Log + Diff)...")
# Log giúp ổn định phương sai, Diff giúp loại bỏ xu hướng
series = np.log(df["usd_index"]).diff().dropna()

# 3. Export
series.to_csv(output_path)
print(f" Preprocessed series saved to {output_path}")
"""
Giải thích:
- log(110) - log(100) ≈ 10% (tỷ lệ thay đổi)
- Dữ liệu 'dừng' (stationary) giúp các model như ARIMA/LSTM dự báo chính xác hơn.
"""
