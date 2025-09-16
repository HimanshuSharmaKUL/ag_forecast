import json
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from src.config import PROC_DIR, REPORTS_DIR, FIGS_DIR
from src.train import FEATURES, TARGET, latest_proc

def main():
    model = joblib.load("models/model.pkl")
    df = pd.read_parquet(latest_proc()).sort_values("date")
    split_idx = int(len(df)*0.8)
    val_df = df.iloc[split_idx:]
    preds = model.predict(val_df[FEATURES])

    out_csv = REPORTS_DIR / "val_predictions.csv"
    pd.DataFrame({
        "date": val_df["date"].values,
        "true": val_df[TARGET].values,
        "pred": preds,
    }).to_csv(out_csv, index=False)

    # Quick plot (true vs pred)
    plt.figure()
    plt.plot(val_df["date"], val_df[TARGET], label="true")
    plt.plot(val_df["date"], preds, label="pred")
    plt.legend()
    plt.title("Validation: next-day close (true vs pred)")
    fig_path = FIGS_DIR / "val_true_vs_pred.png"
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Wrote {out_csv} and {fig_path}")

if __name__ == "__main__":
    main()
