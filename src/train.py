import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

from src.config import PROC_DIR, MODELS_DIR, REPORTS_DIR
from src.utils import save_json, today_str

PROC_PATTERN = "features_*.parquet"

FEATURES = [
    "lag_1","lag_2","lag_3","lag_5","lag_10",
    "roll_ret_mean_5","roll_ret_mean_10","roll_ret_mean_20",
    "roll_ret_std_5","roll_ret_std_10","roll_ret_std_20",
]
TARGET   = "target_close_t+1"

def latest_proc() -> Path:
    files = sorted(PROC_DIR.glob(PROC_PATTERN))
    if not files:
        raise FileNotFoundError("No processed features found. Run `make features` first.")
    return files[-1]

def train_model(df: pd.DataFrame):
    # Keep one ticker at L0 (or concat; we’ll keep it simple)
    # Time-ordered split: last 20% as validation
    df = df.sort_values("date")
    split_idx = int(len(df)*0.8)
    train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val     = val_df[FEATURES],  val_df[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    metrics = {
        "val_mae": float(mean_absolute_error(y_val, preds)),
        "val_r2": float(r2_score(y_val, preds)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "trained_on": today_str(),
    }
    return model, metrics

def main():
    src = latest_proc()
    df = pd.read_parquet(src)
    model, metrics = train_model(df)

    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(model, model_path)

    metrics_path = REPORTS_DIR / "metrics.json"
    save_json(metrics, metrics_path)

    print(f"Saved model → {model_path}")
    print(f"Saved metrics → {metrics_path}")
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
