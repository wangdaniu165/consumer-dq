"""Download unemployment and delinquency series from FRED."""

import requests

from src.config import FRED_URL, RAW_DIR, RAW_PATH


def download_fred(force: bool = False) -> None:
    """Download the combined FRED CSV (UNRATE, DRALACBS, DRCCLACBS) to data/raw/."""
    if RAW_PATH.exists() and not force:
        print(f"FRED data already exists at {RAW_PATH}, skipping.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FRED_URL} ...")
    resp = requests.get(FRED_URL, stream=True, timeout=120)
    resp.raise_for_status()

    with open(RAW_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {RAW_PATH}")


def download_all(force: bool = False) -> None:
    """Download all datasets (single FRED CSV today)."""
    download_fred(force=force)


if __name__ == "__main__":
    download_all()
