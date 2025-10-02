import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------
# Settings (you can tweak)
# --------------------------
TEST_DAYS = 90          # how many recent days to test on
FORECAST_STEPS = 90     # how many days to forecast into the future
SEASONAL_PERIOD = 7     # weekly seasonality for daily data

# --------------------------
# 1) Load and tidy data
# --------------------------
df = pd.read_csv("sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Ensure continuous daily index (fills missing dates if any)
full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
df = df.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
# Fill tiny gaps via interpolation (safe for small gaps)
df["sales"] = df["sales"].interpolate()

# --------------------------
# 2) Train/Test split
# --------------------------
if len(df) < TEST_DAYS + 30:
    # if the series is short, shrink test window
    TEST_DAYS = max(30, len(df) // 5)

train = df.iloc[:-TEST_DAYS].copy()
test = df.iloc[-TEST_DAYS:].copy()

# --------------------------
# 3) Fit SARIMAX
# --------------------------
# A sensible starting point for daily data:
order = (1, 1, 1)
seasonal_order = (1, 1, 1, SEASONAL_PERIOD)

model = SARIMAX(
    train["sales"],
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

res = model.fit(disp=False)

# --------------------------
# 4) Evaluate on test
# --------------------------
pred = res.get_forecast(steps=len(test))
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int(alpha=0.2)  # 80% band

test_eval = test.copy()
test_eval["yhat"] = pred_mean.values
test_eval["yhat_lower"] = pred_ci.iloc[:, 0].values
test_eval["yhat_upper"] = pred_ci.iloc[:, 1].values

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

mape_val = mape(test_eval["sales"], test_eval["yhat"])
mae_val = mae(test_eval["sales"], test_eval["yhat"])
print(f"=== SARIMAX Test Metrics (last {len(test)} days) ===")
print(f"MAPE: {mape_val:.2f}%")
print(f"MAE : {mae_val:.2f}")

# Plot test vs prediction
plt.figure(figsize=(10,4))
plt.plot(test_eval["date"], test_eval["sales"], label="Actual")
plt.plot(test_eval["date"], test_eval["yhat"], label="SARIMAX")
plt.fill_between(test_eval["date"], test_eval["yhat_lower"], test_eval["yhat_upper"], alpha=0.2, label="80% CI")
plt.title(f"SARIMAX — Test Forecast (last {len(test)} days)")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
plt.savefig("sarimax_test.png", dpi=150, bbox_inches="tight")
print("Saved plot: sarimax_test.png")

# --------------------------
# 5) Forecast into future
# --------------------------
future_fc = res.get_forecast(steps=FORECAST_STEPS)
future_mean = future_fc.predicted_mean
future_ci = future_fc.conf_int(alpha=0.2)

last_date = df["date"].max()
future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_STEPS, freq="D")

future = pd.DataFrame({
    "date": future_idx,
    "yhat": future_mean.values,
    "yhat_lower": future_ci.iloc[:, 0].values,
    "yhat_upper": future_ci.iloc[:, 1].values
})

# Plot history + future
plt.figure(figsize=(10,4))
plt.plot(df["date"], df["sales"], label="History")
plt.plot(future["date"], future["yhat"], label="SARIMAX Forecast")
plt.fill_between(future["date"], future["yhat_lower"], future["yhat_upper"], alpha=0.2, label="80% CI")
plt.title(f"SARIMAX — Future Forecast ({FORECAST_STEPS} days)")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
plt.savefig("sarimax_future.png", dpi=150, bbox_inches="tight")
print("Saved plot: sarimax_future.png")

# --------------------------
# 6) Save dashboard-friendly CSV
# --------------------------
# Combine history + future into one long table
hist = df[["date", "sales"]].rename(columns={"sales": "value"})
hist["model"] = "Actual"

fut = future.rename(columns={"yhat": "value"})
fut["model"] = "SARIMAX"

# keep CI columns on the forecast rows
hist["yhat_lower"] = np.nan
hist["yhat_upper"] = np.nan

out = pd.concat([hist, fut], ignore_index=True)
out.to_csv("sarimax_forecast.csv", index=False)
print("Saved CSV: sarimax_forecast.csv")
