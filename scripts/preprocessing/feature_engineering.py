import numpy as np
import pandas as pd

series = pd.read_csv("data/processed/model_ready/usd_series.csv",
                     index_col="date")["usd_index"]

def make_window(data, window=30, horizon=7):
    X, y = [], []
    for i in range(len(data)-window-horizon):
        X.append(data[i:i+window])
        y.append(data[i+window+horizon])
    return np.array(X), np.array(y)

X, y = make_window(series.values)

np.save("data/processed/model_ready/X.npy", X)
np.save("data/processed/model_ready/y.npy", y)


"""
X1 = [r1, r2, ..., r30] → y1 = r38  
X2 = [r2, r3, ..., r31] → y2 = r39  

Dùng 30 ngày quá khứ
để dự đoán 7 ngày trong tương lai

(N, time_steps, features)
→ (N, 30, 1)

"""