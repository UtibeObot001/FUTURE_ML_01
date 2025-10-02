import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Load data ----------
df = pd.read_csv("sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------- Train/Test split ----------
# Use the last 90 days as a simple test window (adjust if your data is short)
TEST_DAYS = 90 if len(df) >= 200 else max(30, len(df)//5)
train = df.iloc[:-TEST_DAYS].copy()
test = df.iloc[-TEST_DAYS:].copy()

# Combine for convenient feature creation without leaking future info
df["sales_shift1"] = df["sales"].shift(1)

# Naïve (yhat = yesterday's actual)
test["yhat_naive"] = df["sales_shift1"].iloc[-TEST_DAYS:].values

# 7-day moving average based on *past info only*
# We compute rolling on sales_shift1 so it only uses data up to t-1
df["ma7"] = df["sales_shift1"].rolling(7, min_periods=1).mean()
test["yhat_ma7"] = df["ma7"].iloc[-TEST_DAYS:].values

# ---------- Metrics ----------
def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mae(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

metrics = {
    "naive": {
        "MAPE": mape(test["sales"], test["yhat_naive"]),
        "MAE": mae(test["sales"], test["yhat_naive"])
    },
    "ma7": {
        "MAPE": mape(test["sales"], test["yhat_ma7"]),
        "MAE": mae(test["sales"], test["yhat_ma7"])
    }
}

print("=== Baseline Metrics (last", TEST_DAYS, "days) ===")
for k, v in metrics.items():
    print(f"{k.upper():<6}  MAPE: {v['MAPE']:.2f}%   MAE: {v['MAE']:.2f}")

# ---------- Plot test window ----------
plt.figure(figsize=(10,4))
plt.plot(test["date"], test["sales"], label="Actual")
plt.plot(test["date"], test["yhat_naive"], label="Naive")
plt.plot(test["date"], test["yhat_ma7"], label="MA(7)")
plt.title(f"Baselines on Test Window (last {TEST_DAYS} days)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("baseline_test.png", dpi=150, bbox_inches="tight")
print("\nSaved plot to baseline_test.png")

# ---------- Export for dashboarding (optional) ----------
out = test[["date", "sales", "yhat_naive", "yhat_ma7"]].rename(columns={"sales":"actual"})
out.to_csv("baseline_test.csv", index=False)
print("Saved results to baseline_test.csv")
