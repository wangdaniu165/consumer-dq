"""Download the NY Fed "Other" 90+ day serious-delinquency series (Equifax).

The New York Fed's Quarterly Report on Household Debt and Credit publishes
"Percent of Balance 90+ Days Delinquent by Loan Type" (report Page 12). One of
its loan types is **Other** — retail + personal installment + other consumer
credit — the closest published proxy for *unsecured consumer (fintech)*
delinquency. It is Equifax-based and **not on FRED**, unlike the FFIEC series
(`DROCLACBS` etc.) used elsewhere in this project.

Run:  python -m src.download_nyfed
Output: ``data/raw/NYFED_OTHER_90DPD.csv``  (quarterly, "2003Q1"..latest)

The report URL is quarterly and changes; the script auto-discovers the latest
report by probing the current quarter and stepping back.
"""

import io
import datetime as dt

import openpyxl
import pandas as pd
import requests

from src.config import RAW_DIR

BASE_URL = ("https://www.newyorkfed.org/medialibrary/interactives/"
            "householdcredit/data/xls/HHD_C_Report_{q}.xlsx")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# "Page 12 Data" column order for the 90+ day delinquency-by-loan-type table.
COLUMNS = ["MORTGAGE", "HELOC", "AUTO", "CC", "STUDENT_LOAN", "OTHER", "ALL"]
SERIES_ID = "NYFED_OTHER_90DPD"


def _quarters_back(n: int = 8) -> list[str]:
    """Return the last `n` quarters as 'YYYYqQ' strings, most recent first."""
    today = dt.date.today()
    yy, q = today.year, (today.month - 1) // 3 + 1
    out = []
    for _ in range(n):
        out.append(f"{yy}q{q}")
        q -= 1
        if q == 0:
            q, yy = 4, yy - 1
    return out


def _fetch_latest() -> tuple[str, bytes]:
    """Probe recent quarters until an HHDC report downloads; return (url, bytes).

    A missing quarter returns a 302 → HTML page (status 200 after redirect), so we
    verify the payload is a real workbook via the xlsx zip magic bytes ``PK``.
    """
    for q in _quarters_back():
        url = BASE_URL.format(q=q)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
        except requests.RequestException:
            continue
        if resp.status_code == 200 and resp.content[:2] == b"PK":
            return url, resp.content
    raise RuntimeError("Could not locate a NY Fed HHDC report (no valid xlsx).")


def _parse(content: bytes) -> pd.DataFrame:
    """Extract the 'Other' 90+ day delinquency column from the workbook."""
    wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True, read_only=True)
    ws = wb["Page 12 Data"]
    rows = list(ws.iter_rows(values_only=True))

    records = []
    for row in rows[4:]:
        if row is None or row[0] is None:
            continue
        label = str(row[0]).strip()
        if ":" not in label:  # skip title/footnote rows
            continue
        yy, _, qn = label.partition(":")
        if not (yy.isdigit() and qn.startswith("Q")):
            continue
        year = 2000 + int(yy)
        quarter = int(qn[1])
        month = (quarter - 1) * 3 + 1
        date = pd.Timestamp(year=year, month=month, day=1)
        other = row[1 + COLUMNS.index("OTHER")]
        records.append((date, float(other)))

    df = pd.DataFrame(records, columns=["observation_date", SERIES_ID])
    return df.sort_values("observation_date").reset_index(drop=True)


def download_other(force: bool = False) -> pd.DataFrame:
    """Download and persist the NY Fed 'Other' series to data/raw."""
    out_path = RAW_DIR / f"{SERIES_ID}.csv"
    if out_path.exists() and not force:
        print(f"{out_path} already exists, skipping.")
        return pd.read_csv(out_path, parse_dates=["observation_date"])

    url, content = _fetch_latest()
    df = _parse(content)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Downloaded {url}")
    print(f"Wrote {out_path} — {len(df)} rows, "
          f"{df['observation_date'].min().date()} .. {df['observation_date'].max().date()}, "
          f"latest {SERIES_ID} = {df[SERIES_ID].iloc[-1]:.2f}%")
    return df


if __name__ == "__main__":
    download_other()
