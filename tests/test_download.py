import pandas as pd

from src import download


def test_download_series_writes_csv(monkeypatch, tmp_path):
    csv = "observation_date,UNRATE\n2020-01-01,3.5\n"
    monkeypatch.setattr(download, "RAW_PATHS", {"UNRATE": tmp_path / "UNRATE.csv"})
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)

    class Resp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            return [csv.encode()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(download.requests, "get", lambda *a, **k: Resp())
    download.download_series("UNRATE", force=True)

    df = pd.read_csv(tmp_path / "UNRATE.csv")
    assert list(df.columns) == ["observation_date", "UNRATE"]


def test_download_skips_when_cached(monkeypatch, tmp_path):
    cached = tmp_path / "UNRATE.csv"
    cached.write_text("observation_date,UNRATE\n2020-01-01,3.5\n")

    monkeypatch.setattr(download, "RAW_PATHS", {"UNRATE": cached})
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)

    called = {"n": 0}

    def fake_get(*a, **k):
        called["n"] += 1
        raise AssertionError("should not re-download when cached")

    monkeypatch.setattr(download.requests, "get", fake_get)
    download.download_series("UNRATE", force=False)  # skip, no requests.get
    assert called["n"] == 0
