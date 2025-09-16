import json
from datetime import datetime
from pathlib import Path

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def today_str():
    return datetime.utcnow().strftime("%Y-%m-%d")
