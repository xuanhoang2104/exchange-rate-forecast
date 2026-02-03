import pandas as pd
import os

# 1. Load Data
# Sử dụng đường dẫn tương đối để linh hoạt hơn
raw_path = r"D:\FPT\ki 7\DAT\exchange-rate\exchange-rate\data\raw\exchange_rate_to_usd.csv"
output_path = r"D:\FPT\ki 7\DAT\exchange-rate\exchange-rate\data\processed\merged\cleaned.csv"

# Đảm bảo thư mục tồn tại
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df = pd.read_csv(raw_path, parse_dates=["date"])
df = df.set_index("date")
df = df.sort_index()

# 2. Xử lý Missing Values
# Sử dụng Interpolate (Nội suy tuyến tính) cho Time Series
# Giúp lấp đầy các khoảng trống dữ liệu một cách tự nhiên
df = df.interpolate(method='linear')

# Backfill cho những giá trị NaN ở đầu chuỗi (nếu có)
df = df.bfill()

# 3. Export
df.to_csv(output_path)
print(f" Data cleaned and saved to {output_path}")

