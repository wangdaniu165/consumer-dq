"""Download unemployment and delinquency series from FRED (one CSV per series)."""

import requests

from src.config import FRED_URL_TEMPLATE, RAW_DIR, RAW_PATHS, SERIES_IDS


def download_series(series_id: str, force: bool = False) -> None:
    """Download one FRED series CSV to data/raw/<id>.csv."""
    path = RAW_PATHS[series_id]
    if path.exists() and not force:
        print(f"{series_id} already exists at {path}, skipping.")
        return

    url = FRED_URL_TEMPLATE.format(id=series_id)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {series_id} -> {path}")


def download_all(force: bool = False) -> None:
    """Download all configured FRED series."""
    for sid in SERIES_IDS:
        download_series(sid, force=force)


if __name__ == "__main__":
    download_all()
