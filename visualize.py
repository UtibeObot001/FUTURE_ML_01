import pandas as pd
import matplotlib.pyplot as plt

# 1) Load data
df = pd.read_csv("sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2) Quick checks
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values per column:\n", df.isna().sum())

# 3) Plot line chart (history)
plt.figure(figsize=(10,4))
plt.plot(df["date"], df["sales"], label="Daily sales")
plt.title("Daily Sales — History")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()

# Save to PNG (so you can open it easily)
plt.savefig("sales_line.png", dpi=150, bbox_inches="tight")
print("\nSaved chart to sales_line.png")
