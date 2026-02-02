import pandas as pd

df = pd.read_csv("data/raw/exchange_rate_to_usd.csv",
                 parse_dates=["date"])
df = df.set_index("date")
df = df.sort_index()
df.to_csv("data/processed/merged/cleaned.csv")
