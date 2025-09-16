import yfinance as yf
import pandas as pd
from src.config import RAW_DIR, TICKERS, START_DATE
from src.utils import today_str

def fetch_yahoo(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, progress=True, auto_adjust=False)
    df["ticker"] = ticker
    df = df.reset_index()
    df = df.rename(columns=str.lower)
    df.columns = [col[0] for col in df.columns] #caution: multiindex header ko single index header me convert kiya hai
    
    
    return df[["date", "ticker", "open", "high", "low", "close", "adj close", "volume"]]

def main():
    frames = []
    for t in TICKERS:
        frames.append(fetch_yahoo(t, START_DATE))
    data = pd.concat(frames).sort_values(["ticker", "date"])
    out = RAW_DIR / f"prices_yahoo_{today_str()}.csv"
    data.to_csv(out, index=False)
    print(f"Wrote {out}, rows={len(data)}")

if __name__ == "__main__":
    main()
