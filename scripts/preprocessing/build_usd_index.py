import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/merged/cleaned.csv",
                 index_col="date", parse_dates=True)

major = [
    'euro_to_usd',
    'japanese_yen_to_usd',
    'uk_pound_to_usd',
    'swiss_franc_to_usd',
    'australian_dollar_to_usd',
    'canadian_dollar_to_usd',
    'chinese_yuan_to_usd'
]

usd_index = df[major].mean(axis=1)
usd_index = usd_index.to_frame("usd_index")
usd_index.to_csv("data/processed/usd/usd_index.csv")
