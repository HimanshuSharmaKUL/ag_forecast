import pandas as pd
from pathlib import Path
from src.config import RAW_DIR, PROC_DIR
from src.utils import today_str

RAW_PATTERN = "prices_yahoo_*.csv"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Use adjusted close as target
    df = df.sort_values(["ticker", "date"])
    df["return_1d"] = df.groupby("ticker")["adj close"].pct_change()
    # Simple lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"] = df.groupby("ticker")["adj close"].shift(lag)
    # Rolling stats on returns
    for w in [5, 10, 20]:
        df[f"roll_ret_mean_{w}"] = df.groupby("ticker")["return_1d"].rolling(w).mean().reset_index(level=0, drop=True)
        df[f"roll_ret_std_{w}"]  = df.groupby("ticker")["return_1d"].rolling(w).std().reset_index(level=0, drop=True)
    # Next-day target (predict tomorrow close or return)
    df["target_close_t+1"] = df.groupby("ticker")["adj close"].shift(-1)
    # Drop rows with NA after feature creation
    df = df.dropna().reset_index(drop=True)
    return df

def latest_raw() -> Path:
    files = sorted(RAW_DIR.glob(RAW_PATTERN))
    if not files:
        raise FileNotFoundError("No raw data found. Run `make ingest` first.")
    return files[-1]

def main():
    src = latest_raw() #pickup the latest raw file ingested (in previous step)
    df = pd.read_csv(src, parse_dates=["date"])
    feat = build_features(df)
    print(feat.head())
    out = PROC_DIR / f"features_{today_str()}.parquet"
    feat.to_parquet(out, index=False)
    print(f"Wrote {out}, rows={len(feat)}")

if __name__ == "__main__":
    main()
