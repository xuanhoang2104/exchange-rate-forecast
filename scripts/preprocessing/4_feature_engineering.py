import numpy as np
import pandas as pd
import os

# 1. Config
input_path = "data/processed/model_ready/usd_series.csv"
output_dir = "data/processed/model_ready"
os.makedirs(output_dir, exist_ok=True)

series = pd.read_csv(input_path, index_col="date")["usd_index"]

def make_window(data, window=30, horizon=7):
    """
    Cắt Sliding Window cho Model
    window: số ngày quá khứ
    horizon: khoảng cách đến ngày cần dự báo trong tương lai
    """
    X, y = [], []
    for i in range(len(data)-window-horizon):
        X.append(data[i:i+window])
        # Lấy nhãn là giá trị sau 'horizon' bước
        y.append(data[i+window+horizon])
    return np.array(X), np.array(y)

# 2. Process
print(f" Engineering features with window=30, horizon=7...")
X, y = make_window(series.values)

# 3. Save as .npy (Numpy format for fast loading in Training)
np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)

print(f" Features saved: X shape {X.shape}, y shape {y.shape}")

"""
Slide Window Flow:
X1 = [r1, r2, ..., r30] -> y1 = r37/r38 (tùy theo mốc thời gian)
Dùng 30 ngày liên tiếp để dự báo thay đổi vào 7 ngày sau.
"""
