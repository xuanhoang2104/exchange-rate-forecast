import pandas as pd
import numpy as np
import os

# 1. Config
input_path = "data/processed/merged/cleaned.csv"
output_path = "data/processed/usd/usd_index.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df = pd.read_csv(input_path, index_col="date", parse_dates=True)

major = [
    'euro_to_usd',
    'japanese_yen_to_usd',
    'uk_pound_to_usd',
    'swiss_franc_to_usd',
    'australian_dollar_to_usd',
    'canadian_dollar_to_usd',
    'chinese_yuan_to_usd'
]

# 2. Build Index
print(f" Building USD Index from {len(major)} major currencies...")
usd_index = df[major].mean(axis=1)
usd_index = usd_index.to_frame("usd_index")

# 3. Save
usd_index.to_csv(output_path)
print(f" USD Index saved to {output_path}")

