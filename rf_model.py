import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# -------- settings --------
TEST_DAYS = 90
FORECAST_STEPS = 90
RANDOM_STATE = 42

# -------- load data --------
df = pd.read_csv("sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ensure continuous daily dates
full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
df = df.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
df["sales"] = df["sales"].interpolate()

# -------- split train/test --------
if len(df) < TEST_DAYS + 30:
    TEST_DAYS = max(30, len(df)//5)

train = df.iloc[:-TEST_DAYS].copy()
test = df.iloc[-TEST_DAYS:].copy()

# -------- feature maker --------
def make_features(frame: pd.DataFrame):
    X = frame.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["dow"] = X["date"].dt.dayofweek
    X["week"] = X["date"].dt.isocalendar().week.astype(int)

    # lags
    for lag in [1,7,14,28]:
        X[f"lag_{lag}"] = X["sales"].shift(lag)
    # rolling means (use min_periods so the start doesn’t break)
    X["roll7"] = X["sales"].rolling(7, min_periods=1).mean()
    X["roll28"] = X["sales"].rolling(28, min_periods=1).mean()

    # drop rows created by lags that are NaN
    X = X.dropna().reset_index(drop=True)
    return X

feat = make_features(df)
# align split after features
split_idx = len(feat) - TEST_DAYS
train_feat = feat.iloc[:split_idx].copy()
test_feat  = feat.iloc[split_idx:].copy()

FEATURES = [c for c in train_feat.columns if c not in ["date","sales"]]

# -------- train model --------
rf = RandomForestRegressor(
    n_estimators=400,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(train_feat[FEATURES], train_feat["sales"])

# -------- evaluate on test --------
pred_test = rf.predict(test_feat[FEATURES])

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

print(f"=== RF Test Metrics (last {TEST_DAYS} days) ===")
print(f"MAPE: {mape(test_feat['sales'], pred_test):.2f}%")
print(f"MAE : {mae(test_feat['sales'], pred_test):.2f}")

# plot test fit
plt.figure(figsize=(10,4))
plt.plot(test_feat["date"], test_feat["sales"], label="Actual")
plt.plot(test_feat["date"], pred_test, label="RF")
plt.title(f"RandomForest — Test Forecast (last {TEST_DAYS} days)")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
plt.savefig("rf_test.png", dpi=150, bbox_inches="tight")
print("Saved: rf_test.png")

# -------- forecast next 90 days --------
# we forecast iteratively: add future rows, build features using last known/pred values
hist = df.copy()
future_rows = []
last_date = hist["date"].max()

for step in range(FORECAST_STEPS):
    next_date = last_date + pd.Timedelta(days=1)
    last_date = next_date

    # temporary frame including a placeholder next day
    tmp = pd.concat([hist, pd.DataFrame([{"date": next_date, "sales": np.nan}])], ignore_index=True)

    # build lags/rolls from tmp
    tmp_feat = make_features(tmp)

    # the last row in tmp_feat corresponds to next_date after feature construction
    row = tmp_feat.iloc[[-1]][FEATURES]  # keep as DataFrame
    yhat = rf.predict(row)[0]

    # write prediction back so next step can use it as a lag
    hist.loc[len(hist)] = {"date": next_date, "sales": yhat}
    future_rows.append({"date": next_date, "yhat": yhat})

future = pd.DataFrame(future_rows)

# plot future forecast
plt.figure(figsize=(10,4))
plt.plot(df["date"], df["sales"], label="History")
plt.plot(future["date"], future["yhat"], label="RF Forecast")
plt.title(f"RandomForest — Future Forecast ({FORECAST_STEPS} days)")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
plt.savefig("rf_future.png", dpi=150, bbox_inches="tight")
print("Saved: rf_future.png")

# -------- save CSV (Power BI–friendly) --------
rf_out = future.rename(columns={"yhat": "value"})
rf_out["model"] = "RandomForest"
rf_out["yhat_lower"] = np.nan
rf_out["yhat_upper"] = np.nan
rf_out.to_csv("rf_forecast.csv", index=False)
print("Saved: rf_forecast.csv")
