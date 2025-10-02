import pandas as pd
from pathlib import Path

sarimax_path = Path("sarimax_forecast.csv")   # Actual + SARIMAX + CI
rf_path = Path("rf_forecast.csv")             # RandomForest (no CI)

if not sarimax_path.exists():
    raise FileNotFoundError("sarimax_forecast.csv not found. Run sarimax_model.py first.")
if not rf_path.exists():
    raise FileNotFoundError("rf_forecast.csv not found. Run rf_model.py first.")

sarimax = pd.read_csv(sarimax_path, parse_dates=["date"])
rf = pd.read_csv(rf_path, parse_dates=["date"])

# Ensure CI columns exist in both
for col in ["yhat_lower","yhat_upper"]:
    if col not in sarimax.columns:
        sarimax[col] = pd.NA
    if col not in rf.columns:
        rf[col] = pd.NA

actual = sarimax[sarimax["model"]=="Actual"].copy()
srx    = sarimax[sarimax["model"]=="SARIMAX"].copy()

cols = ["date","model","value","yhat_lower","yhat_upper"]
combined = pd.concat([actual[cols], srx[cols], rf[cols]], ignore_index=True).sort_values("date")
combined.to_csv("combined_forecasts.csv", index=False)
print("Saved: combined_forecasts.csv")
