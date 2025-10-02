import pandas as pd
import numpy as np

np.random.seed(123)

# Daily dates: 2022-01-01 to 2025-09-22
dates = pd.date_range("2022-01-01", "2025-09-22", freq="D")
n = len(dates)

# Trend + weekly (weekends up) + yearly + promo spikes + noise
trend = np.linspace(100, 210, n)
weekly = 12*np.sin(2*np.pi * dates.dayofweek / 7) + 6*(dates.dayofweek >= 5).astype(int)
yearly = 18*np.sin(2*np.pi * dates.dayofyear / 365.25)
promos = np.random.binomial(1, 0.03, n)
promo_effect = promos * np.random.uniform(25, 70, n)
noise = np.random.normal(0, 9, n)

sales = trend + weekly + yearly + promo_effect + noise
sales = np.clip(sales, 0, None)

df = pd.DataFrame({"date": dates, "sales": np.round(sales, 2)})
df.to_csv("sales.csv", index=False)
print("Created sales.csv with", len(df), "rows.")
