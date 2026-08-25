import pandas as pd

from src import download


def test_download_fred_parses_columns(monkeypatch, tmp_path):
    csv = "DATE,UNRATE,DRALACBS,DRCCLACBS\n2020-01-01,3.5,1.5,2.5\n"

    monkeypatch.setattr(download, "RAW_PATH", tmp_path / "fred.csv")
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
    download.download_fred(force=True)

    df = pd.read_csv(download.RAW_PATH)
    assert list(df.columns) == ["DATE", "UNRATE", "DRALACBS", "DRCCLACBS"]


def test_download_skips_when_cached(monkeypatch, tmp_path):
    cached = tmp_path / "fred.csv"
    cached.write_text("DATE,UNRATE,DRALACBS,DRCCLACBS\n2020-01-01,3.5,1.5,2.5\n")

    monkeypatch.setattr(download, "RAW_PATH", cached)
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)

    called = {"n": 0}

    def fake_get(*a, **k):
        called["n"] += 1
        raise AssertionError("should not re-download when cached")

    monkeypatch.setattr(download.requests, "get", fake_get)
    download.download_fred(force=False)  # should skip without calling requests.get
    assert called["n"] == 0
