from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGS_DIR = REPORTS_DIR / "figures"

for d in [RAW_DIR, PROC_DIR, MODELS_DIR, REPORTS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Which tickers/series to pull at Level 0
TICKERS = ["ZC=F"]  # start with Corn; add "ZW=F", "ZS=F" later
START_DATE = "2010-01-01"
