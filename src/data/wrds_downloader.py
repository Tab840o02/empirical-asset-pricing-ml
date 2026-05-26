"""
src/data/wrds_downloader.py
============================
WRDS raw data download functions for the GKX (2020) replication project.

All functions accept a ``wrds.Connection`` instance and write one Parquet
file per table to ``data/raw/``.  Open **one** connection per session and
pass it into every function — do not re-open per call.

Quick-start
-----------
Set your WRDS username once per shell session (never hard-code it):

    PowerShell:  $env:WRDS_USER = "your_username"
    bash/zsh:    export WRDS_USER="your_username"

Then run the full download pipeline:

    python -m src.data.wrds_downloader

The first time you run this, ``wrds`` will prompt for your password and
cache it in a local pgpass file.  Every subsequent run is non-interactive.

Tables downloaded
-----------------
  CRSP
    crsp_monthly.parquet         Monthly stock file (msf)
    crsp_names.parquet           Historical names / identifiers (msenames)
    crsp_delistings.parquet      Delisting returns and codes (mse)
    crsp_daily_index.parquet     Value- and equal-weighted market returns (dsi)
    crsp_daily_{year}.parquet    Daily stock file, one file per year (dsf)

  Compustat
    compustat_annual.parquet     Annual fundamentals (funda)
    compustat_quarterly.parquet  Quarterly fundamentals (fundq)
    compustat_security.parquet   Security-level identifiers & CUSIPs (security)
    compustat_company.parquet    Company-level identifiers & SIC codes (company)

  Benchmarks (free — no WRDS needed)
    ff_factors.parquet           FF 5-factor + RF monthly returns
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    ACTIVE_END_DATE,
    COMPUSTAT_START,
    RAW_DIR,
    SAMPLE_START,
    WRDS_USER,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def open_connection():
    """
    Open and return a ``wrds.Connection``.

    Reads the username from the ``WRDS_USER`` environment variable (set in
    ``src/config.py`` from ``os.environ``).  The password is read from the
    local pgpass cache; if no cache exists, ``wrds`` will prompt once.
    """
    import wrds  # imported here so the rest of the module works without wrds installed

    if WRDS_USER is None:
        raise EnvironmentError(
            "WRDS_USER environment variable is not set.  "
            "Run:  $env:WRDS_USER = 'your_username'  (PowerShell)  "
            "or    export WRDS_USER='your_username'   (bash/zsh)"
        )
    log.info("Connecting to WRDS as user '%s' …", WRDS_USER)
    return wrds.Connection(wrds_username=WRDS_USER)


# ---------------------------------------------------------------------------
# CRSP
# ---------------------------------------------------------------------------


def download_crsp_monthly(conn, start_date: str = SAMPLE_START, output_dir: Path = RAW_DIR) -> Path:
    """
    Download ``crsp.msf`` — the CRSP Monthly Stock File.

    Columns: permno, date, ret, retx, prc, shrout, vol, cfacshr
    """
    log.info("Downloading CRSP Monthly Stock File …")
    sql = f"""
        SELECT
            permno,
            date,
            ret,
            retx,
            prc,
            shrout,
            vol,
            cfacshr
        FROM crsp.msf
        WHERE date >= '{start_date}'
        ORDER BY permno, date
    """
    df = conn.raw_sql(sql, date_cols=["date"])
    out = output_dir / "crsp_monthly.parquet"
    df.to_parquet(out, index=False)
    log.info("  → %s  (%d rows)", out.name, len(df))
    return out


def download_crsp_names(conn, output_dir: Path = RAW_DIR) -> Path:
    """
    Download ``crsp.msenames`` — CRSP historical names, identifiers, and
    exchange/share-class codes.

    Key columns for the CUSIP link:
        permno, permco, ncusip (8-char, no check digit), ticker, comnam,
        shrcd, exchcd, namedt, nameendt
    """
    log.info("Downloading CRSP Names (msenames) …")
    sql = """
        SELECT
            permno,
            permco,
            cusip,
            ncusip,
            ticker,
            comnam,
            shrcd,
            exchcd,
            namedt,
            nameendt
        FROM crsp.msenames
        ORDER BY permno, namedt
    """
    df = conn.raw_sql(sql, date_cols=["namedt", "nameendt"])
    out = output_dir / "crsp_names.parquet"
    df.to_parquet(out, index=False)
    log.info("  → %s  (%d rows)", out.name, len(df))
    return out


def download_crsp_delistings(conn, output_dir: Path = RAW_DIR) -> Path:
    """
    Download delisting records from ``crsp.mse`` (CRSP Monthly Stock Events).

    Only rows with a non-null ``dlstcd`` are returned; these represent the
    securities that were removed from the exchange and are essential for
    correcting survivorship bias (Shumway 1997).

    Columns: permno, date, dlstcd, dlret, dlpdt
    """
    log.info("Downloading CRSP Delistings (mse) …")
    sql = """
        SELECT
            permno,
            date,
            dlstcd,
            dlret,
            dlpdt
        FROM crsp.mse
        WHERE dlstcd IS NOT NULL
        ORDER BY permno, date
    """
    df = conn.raw_sql(sql, date_cols=["date", "dlpdt"])
    out = output_dir / "crsp_delistings.parquet"
    df.to_parquet(out, index=False)
    log.info("  → %s  (%d rows)", out.name, len(df))
    return out


def download_crsp_daily_index(conn, start_date: str = SAMPLE_START, output_dir: Path = RAW_DIR) -> Path:
    """
    Download ``crsp.dsi`` — CRSP Daily Stock Market Indices.

    Used as the market return series when computing beta and idiosyncratic
    volatility from daily stock returns.

    Columns: date, vwretd (value-weighted, with distributions),
             ewretd (equal-weighted)
    """
    log.info("Downloading CRSP Daily Market Index (dsi) …")
    sql = f"""
        SELECT
            date,
            vwretd,
            ewretd
        FROM crsp.dsi
        WHERE date >= '{start_date}'
        ORDER BY date
    """
    df = conn.raw_sql(sql, date_cols=["date"])
    out = output_dir / "crsp_daily_index.parquet"
    df.to_parquet(out, index=False)
    log.info("  → %s  (%d rows)", out.name, len(df))
    return out


def download_crsp_daily_year(conn, year: int, output_dir: Path = RAW_DIR) -> Path:
    """
    Download one calendar year of ``crsp.dsf`` — the CRSP Daily Stock File.

    Always pull year-by-year; the full daily file contains >100 million rows
    and will time out or exhaust memory in a single query.

    Columns: permno, date, ret, vol, prc
    """
    log.info("Downloading CRSP Daily Stock File — year %d …", year)
    sql = f"""
        SELECT
            permno,
            date,
            ret,
            vol,
            prc
        FROM crsp.dsf
        WHERE date >= '{year}-01-01'
          AND date <= '{year}-12-31'
        ORDER BY permno, date
    """
    df = conn.raw_sql(sql, date_cols=["date"])
    out = output_dir / f"crsp_daily_{year}.parquet"
    df.to_parquet(out, index=False)
    log.info("  → %s  (%d rows)", out.name, len(df))
    return out


def download_crsp_daily_range(
    conn,
    start_year: int = 1957,
    end_year: int | None = None,
    output_dir: Path = RAW_DIR,
) -> list[Path]:
    """
    Download all years of CRSP daily data from ``start_year`` through
    ``end_year`` (inclusive).  If ``end_year`` is None, uses the current year.
    """
    import datetime

    if end_year is None:
        end_year = datetime.date.today().year

    paths = []
    for year in range(start_year, end_year + 1):
        out = output_dir / f"crsp_daily_{year}.parquet"
        if out.exists():
            log.info("  Skipping CRSP daily %d — file already exists.", year)
            paths.append(out)
            continue
        paths.append(download_crsp_daily_year(conn, year, output_dir))
    return paths


# ---------------------------------------------------------------------------
# Compustat
# ---------------------------------------------------------------------------

# Compustat Annual columns needed for the 94 GKX characteristics
_FUNDA_COLS = """
    gvkey, datadate, fyear, indfmt, datafmt, popsrc, consol,
    -- Identifiers
    sich, naicsh,
    -- Size / shares
    at, lt, ceq, seq, csho, prcc_f, ajex, mkvalt,
    -- Preferred stock (needed for book equity)
    pstkl, pstkrv, pstk, txditc, txdb, itcb,
    -- Income statement
    revt, sale, cogs, xsga, xrd, dp, ib, ni, ebit, ebitda, pi, txt, mii,
    -- Cash flow
    oancf, capx, dv, fincf, ivncf,
    -- Balance sheet — assets
    act, che, rect, invt, ppent, ppegt, intan, ao, ivao,
    -- Balance sheet — liabilities
    lct, dlc, dltt, dd1, ap, txp, ob,
    -- Equity / other
    re, prstkc, sstk, dvc, dvp, dvt,
    -- Convertible debt (used for convind — Valta 2012)
    dcvt,
    -- Accruals
    xacc, recch, invch,
    -- Employee count (for hire = Δemp/lag(emp) — Belo et al. 2014)
    emp
"""

# Compustat Quarterly columns
_FUNDQ_COLS = """
    gvkey, datadate, fyearq, fqtr, indfmt, datafmt, popsrc, consol,
    -- Quarterly fundamentals
    atq, ltq, ceqq, seqq, cshoq, prccq, ajexq,
    ibq, saleq, niq, dpq, xrdq, oancfy, capsq,
    actq, cheq, rectq, invtq, ppentq, lctq, dlcq, dlttq,
    txdbq, txditcq,
    -- Total income taxes (used for chtx — Thomas & Zhang 2011)
    txtq,
    -- For SUE (standardised unexpected earnings)
    epspxq, epspiq,
    -- Earnings announcement date (for ear — Kishore et al. 2008)
    rdq
"""


def download_compustat_annual(
    conn,
    start_date: str = COMPUSTAT_START,
    output_dir: Path = RAW_DIR,
) -> Path:
    """
    Download ``comp.funda`` — Compustat Annual Fundamentals.

    Applies the standard four-filter combination that eliminates duplicate
    and non-industrial rows from Compustat.
    """
    log.info("Downloading Compustat Annual (funda) …")
    # Build a clean column list (strip comments and whitespace)
    cols = ", ".join(
        c.strip()
        for c in _FUNDA_COLS.replace("\n", ",").split(",")
        if c.strip() and not c.strip().startswith("--")
    )
    sql = f"""
        SELECT {cols}
        FROM comp.funda
        WHERE indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
          AND datadate >= '{start_date}'
        ORDER BY gvkey, datadate
    """
    df = conn.raw_sql(sql, date_cols=["datadate"])
    out = output_dir / "compustat_annual.parquet"
    df.to_parquet(out, index=False)
    log.info("  → %s  (%d rows)", out.name, len(df))
    return out


def download_compustat_quarterly(
    conn,
    start_date: str = COMPUSTAT_START,
    output_dir: Path = RAW_DIR,
) -> Path:
    """
    Download ``comp.fundq`` — Compustat Quarterly Fundamentals.

    Used for quarterly-updated characteristics: roaq, sue, chtx, rsup.
    Applies the same four-filter combination as the annual download.
    """
    log.info("Downloading Compustat Quarterly (fundq) …")
    cols = ", ".join(
        c.strip()
        for c in _FUNDQ_COLS.replace("\n", ",").split(",")
        if c.strip() and not c.strip().startswith("--")
    )
    sql = f"""
        SELECT {cols}
        FROM comp.fundq
        WHERE indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
          AND datadate >= '{start_date}'
        ORDER BY gvkey, datadate
    """
    df = conn.raw_sql(sql, date_cols=["datadate"])
    out = output_dir / "compustat_quarterly.parquet"
    df.to_parquet(out, index=False)
    log.info("  → %s  (%d rows)", out.name, len(df))
    return out


def download_compustat_security(conn, output_dir: Path = RAW_DIR) -> Path:
    """
    Download ``comp.security`` — Compustat Security-level identifiers.

    This is the primary source of CUSIPs on the Compustat side.  Each row
    is one GVKEY–IID pair (i.e. one share class / listing).

    Key columns:
        gvkey  — company identifier (Compustat)
        iid    — issue identifier (security within the company)
        cusip  — 9-character CUSIP (first 8 chars = issuer + issue,
                 last char = check digit)
        tic    — ticker symbol
        conm   — company name
        idbeg  — date this CUSIP became valid
        idend  — date this CUSIP expired (NULL = still active)
    """
    log.info("Downloading Compustat Security (security) …")
    sql = f"""
        SELECT
            gvkey,
            iid,
            cusip,
            tic,
            exchg,
            tpci
        FROM comp.security
        ORDER BY gvkey, iid
    """
    df = conn.raw_sql(sql)
    out = output_dir / "compustat_security.parquet"
    df.to_parquet(out, index=False)
    log.info("  → %s  (%d rows)", out.name, len(df))
    return out


def download_compustat_company(conn, output_dir: Path = RAW_DIR) -> Path:
    """
    Download ``comp.company`` — Compustat Company-level metadata.

    Used to obtain industry codes (SIC, NAICS, GICS) at the company level
    for industry-adjusted characteristics.
    """
    log.info("Downloading Compustat Company (company) …")
    sql = """
        SELECT
            gvkey,
            conm,
            sic,
            naics,
            gind,
            gsector,
            fic,
            loc,
            incorp
        FROM comp.company
        ORDER BY gvkey
    """
    df = conn.raw_sql(sql)
    out = output_dir / "compustat_company.parquet"
    df.to_parquet(out, index=False)
    log.info("  → %s  (%d rows)", out.name, len(df))
    return out


# ---------------------------------------------------------------------------
# Fama-French factors (free download — no WRDS connection required)
# ---------------------------------------------------------------------------


def download_ff_factors(output_dir: Path = RAW_DIR) -> Path:
    """
    Download monthly Fama-French 5-factor data directly from Kenneth French's
    website using ``requests`` and ``zipfile`` (no pandas_datareader needed).

    Saves a single Parquet file containing:
        Mkt-RF, SMB, HML, RMW, CMA, RF  (all as decimals, not percentages)

    The risk-free rate (RF) in this file is the 1-month T-bill return used
    to convert raw CRSP returns to excess returns for model training.
    """
    import io
    import zipfile
    import requests

    BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    FF3_URL = f"{BASE}/F-F_Research_Data_Factors_CSV.zip"
    FF5_URL = f"{BASE}/F-F_Research_Data_5_Factors_2x3_CSV.zip"

    def _fetch_csv(url: str) -> pd.DataFrame:
        log.info("  Fetching %s …", url.split("/")[-1])
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw = zf.read(csv_name).decode("utf-8", errors="replace")
        # Ken French CSVs have a header block and a footer — parse only the
        # rows where the first column is a 6-digit YYYYMM integer.
        lines = []
        for line in raw.splitlines():
            parts = line.split(",")
            token = parts[0].strip()
            if len(token) == 6 and token.isdigit():
                lines.append(line)
        df = pd.read_csv(
            io.StringIO("\n".join(lines)),
            header=None,
        )
        # First column is the period index (YYYYMM)
        df.columns = range(len(df.columns))
        df[0] = pd.to_datetime(df[0].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
        df = df.set_index(0)
        df.index.name = "date"
        return df

    log.info("Downloading Fama-French factor data from Kenneth French website …")

    ff3 = _fetch_csv(FF3_URL)
    ff3.columns = ["Mkt-RF", "SMB", "HML", "RF"]

    ff5 = _fetch_csv(FF5_URL)
    ff5.columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]

    # Use FF3 for the full history (1926+); fill in RMW, CMA from FF5 where available
    ff = ff3[["Mkt-RF", "SMB", "HML", "RF"]].copy()
    ff = ff.join(ff5[["RMW", "CMA"]], how="left")

    # Convert from percentage to decimal
    ff = ff / 100.0

    out = output_dir / "ff_factors.parquet"
    ff.to_parquet(out)
    log.info("  → %s  (%d months)", out.name, len(ff))
    return out


def download_phase5a_extension(
    conn,
    crsp_start_date: str = "2017-01-01",
    compustat_annual_start: str = "2017-01-01",
    compustat_quarterly_start: str = "2016-01-01",
    daily_start_year: int = 2017,
    output_dir: Path = RAW_DIR,
) -> pd.Timestamp:
    """
    Download the post-2016 extension tables needed for Phase 5a.

    This is a convenience wrapper around the existing table-specific download
    helpers.  It keeps the extension pull explicit without changing the
    canonical Phase 1 workflow.
    """
    latest = get_latest_crsp_date(conn)
    log.info("Downloading Phase 5a extension tables through %s …", latest.date())

    download_crsp_monthly(conn, start_date=crsp_start_date, output_dir=output_dir)
    download_crsp_names(conn, output_dir=output_dir)
    download_crsp_delistings(conn, output_dir=output_dir)
    download_crsp_daily_index(conn, start_date=crsp_start_date, output_dir=output_dir)
    download_crsp_daily_range(conn, start_year=daily_start_year, end_year=latest.year, output_dir=output_dir)

    download_compustat_annual(conn, start_date=compustat_annual_start, output_dir=output_dir)
    download_compustat_quarterly(conn, start_date=compustat_quarterly_start, output_dir=output_dir)
    download_compustat_security(conn, output_dir=output_dir)
    download_compustat_company(conn, output_dir=output_dir)
    download_ff_factors(output_dir=output_dir)

    log.info("Phase 5a extension downloads complete.")
    return latest


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def get_latest_crsp_date(conn) -> pd.Timestamp:
    """
    Query the latest available date in ``crsp.msf``.
    Used to set ``EXT_END`` at runtime without hard-coding a date.
    """
    result = conn.raw_sql("SELECT MAX(date) AS max_date FROM crsp.msf")
    latest = pd.Timestamp(result["max_date"].iloc[0])
    log.info("Latest CRSP monthly date available: %s", latest.date())
    return latest


# ---------------------------------------------------------------------------
# CLI entry-point  (python -m src.data.wrds_downloader)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    conn = open_connection()
    try:
        # --- CRSP ---
        download_crsp_monthly(conn)
        download_crsp_names(conn)
        download_crsp_delistings(conn)
        download_crsp_daily_index(conn)

        latest = get_latest_crsp_date(conn)
        end_year = latest.year

        download_crsp_daily_range(conn, start_year=1957, end_year=end_year)

        # --- Compustat ---
        download_compustat_annual(conn)
        download_compustat_quarterly(conn)
        download_compustat_security(conn)
        download_compustat_company(conn)

    finally:
        conn.close()
        log.info("WRDS connection closed.")

    # --- FF factors (no WRDS needed) ---
    download_ff_factors()

    log.info("Phase 1 downloads complete.  All files written to %s", RAW_DIR)
    sys.exit(0)
