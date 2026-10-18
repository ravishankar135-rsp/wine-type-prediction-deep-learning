"""Download the original public wine-quality datasets into data/."""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
FILES = {
    "redwinequality.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "whitewinequality.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}
for filename, url in FILES.items():
    destination = DATA_DIR / filename
    if destination.exists():
        print(f"Already present: {destination}")
        continue
    print(f"Downloading {filename} ...")
    urlretrieve(url, destination)
    print(f"Saved: {destination}")
print("Dataset setup complete.")
