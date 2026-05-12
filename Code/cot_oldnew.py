"""
cot_oldnew.py — Old-crop / New-crop COT split
==============================================
Produces cot_oldnew.parquet in the COT Database folder.

Schema:
    Commodity, Crop (All / Old / Other), Date,
    Prod Long, Prod Short,
    Swap Long, Swap Short, Swap Spread,
    MM Long, MM Short, MM Spread,
    Other Long, Other Short,
    Non Rep Long, Non Rep Short,
    Total OI, Px

Usage:
    python cot_oldnew.py           # incremental
    python cot_oldnew.py --full    # full rebuild from START_DATE
"""

import argparse
import icepython as ice
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_DIR     = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT\Database")
OUT_PATH   = DB_DIR / "cot_oldnew.parquet"
START_DATE = "2014-01-01"

COMMODITIES = {
    "KC":  {"cot_sym": "KC #COMB-CFTC",      "px_sym": "%KC 1!"},
    "CC":  {"cot_sym": "CC #COMB-CFTC",      "px_sym": "%CC 1!"},
    "SB":  {"cot_sym": "SB #COMB-CFTC",      "px_sym": "%SB 1!"},
    "CT":  {"cot_sym": "CT #COMB-CFTC",      "px_sym": "%CT 1!"},
    "RC":  {"cot_sym": "RC.ICE #COMB-CFTC",  "px_sym": "%RC 1!-ICE"},
    "LCC": {"cot_sym": "C.ICE #COMB-CFTC",   "px_sym": "%C 1!-ICE"},
}

# One set of fields per crop level — three separate calls to keep ICE happy
FIELDS = {
    "All": [
        "Open Interest All Close",
        "Prod Merc Positions Long All Close",
        "Prod Merc Positions Short All Close",
        "Swap Positions Long All Close",
        "Swap Positions Short All Close",
        "Swap Positions Spread All Close",
        "M Money Positions Long All Close",
        "M Money Positions Short All Close",
        "M Money Positions Spread All Close",
        "Other Rept Positions Long All Close",
        "Other Rept Positions Short All Close",
        "Nonrept Positions Long All Close",
        "Nonrept Positions Short All Close",
    ],
    "Old": [
        "Open Interest Old Close",
        "Prod Merc Positions Long Old Close",
        "Prod Merc Positions Short Old Close",
        "Swap Positions Long Old Close",
        "Swap Positions Short Old Close",
        "Swap Positions Spread Old Close",
        "M Money Positions Long Old Close",
        "M Money Positions Short Old Close",
        "M Money Positions Spread Old Close",
        "Other Rept Positions Long Old Close",
        "Other Rept Positions Short Old Close",
        "Nonrept Positions Long Old Close",
        "Nonrept Positions Short Old Close",
    ],
    "Other": [
        "Open Interest Other Close",
        "Prod Merc Positions Long Other Close",
        "Prod Merc Positions Short Other Close",
        "Swap Positions Long Other Close",
        "Swap Positions Short Other Close",
        "Swap Positions Spread Other Close",
        "M Money Positions Long Other Close",
        "M Money Positions Short Other Close",
        "M Money Positions Spread Other Close",
        "Other Rept Positions Long Other Close",
        "Other Rept Positions Short Other Close",
        "Nonrept Positions Long Other Close",
        "Nonrept Positions Short Other Close",
    ],
}

# Clean column names (same order as FIELDS lists above)
CLEAN_COLS = [
    "Total OI",
    "Prod Long", "Prod Short",
    "Swap Long", "Swap Short", "Swap Spread",
    "MM Long", "MM Short", "MM Spread",
    "Other Long", "Other Short",
    "Non Rep Long", "Non Rep Short",
]

FINAL_COLS = [
    "Commodity", "Crop", "Date",
    "Prod Long", "Prod Short",
    "Swap Long", "Swap Short", "Swap Spread",
    "MM Long", "MM Short", "MM Spread",
    "Other Long", "Other Short",
    "Non Rep Long", "Non Rep Short",
    "Total OI", "Px",
]

INT_COLS = [
    "Prod Long", "Prod Short",
    "Swap Long", "Swap Short", "Swap Spread",
    "MM Long", "MM Short", "MM Spread",
    "Other Long", "Other Short",
    "Non Rep Long", "Non Rep Short",
    "Total OI",
]


def _fetch_crop_chunk(cot_sym, crop, start, end):
    fields = FIELDS[crop]
    try:
        data = ice.get_timeseries(cot_sym, fields, granularity="D",
                                  start_date=start, end_date=end)
        rows = list(data)
        if not rows or "Error" in str(rows[0][0]):
            return None
        data_rows = rows[1:]
        if not data_rows:
            return None
        df = pd.DataFrame(data_rows, columns=["Date"] + CLEAN_COLS)
        df["Date"] = pd.to_datetime(df["Date"])
        for c in CLEAN_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Crop"] = crop
        return df.set_index("Date")
    except Exception as e:
        print(f"  ERROR {cot_sym} [{crop}]: {e}")
        return None


def fetch_crop(cot_sym, crop, start, end):
    """Fetch in yearly chunks to avoid ICE row limits."""
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    chunks   = []
    year = start_dt.year
    while True:
        chunk_start = max(start_dt, pd.Timestamp(f"{year}-01-01"))
        chunk_end   = min(end_dt,   pd.Timestamp(f"{year}-12-31"))
        df = _fetch_crop_chunk(cot_sym, crop,
                               chunk_start.strftime("%Y-%m-%d"),
                               chunk_end.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            chunks.append(df)
        year += 1
        if chunk_end >= end_dt:
            break
    if not chunks:
        return None
    return pd.concat(chunks).sort_index()


def fetch_px(px_sym, start, end):
    try:
        data = ice.get_timeseries(px_sym, ["Settle"], granularity="D",
                                  start_date=start, end_date=end)
        rows = list(data)
        if not rows or "Error" in str(rows[0][0]):
            return pd.Series(dtype=float)
        df = pd.DataFrame(rows[1:], columns=["Date", "Px"])
        df["Date"] = pd.to_datetime(df["Date"])
        df["Px"] = pd.to_numeric(df["Px"], errors="coerce")
        return df.set_index("Date")["Px"]
    except Exception as e:
        print(f"  ERROR px {px_sym}: {e}")
        return pd.Series(dtype=float)


def build_commodity(comm, cfg, start, end):
    pieces = []
    for crop in ("All", "Old", "Other"):
        print(f"  [{crop}] {cfg['cot_sym']}...", end=" ", flush=True)
        df = fetch_crop(cfg["cot_sym"], crop, start, end)
        if df is None or df.empty:
            print("no data")
        else:
            print(f"{len(df)} rows")
            pieces.append(df)

    if not pieces:
        return None

    long = pd.concat(pieces).reset_index()
    long["Commodity"] = comm

    print(f"  [Px]  {cfg['px_sym']}...", end=" ", flush=True)
    px = fetch_px(cfg["px_sym"], start, end)
    print(f"{len(px)} rows")
    px_df = px.reset_index()
    px_df.columns = ["Date", "Px"]
    long = long.merge(px_df, on="Date", how="left")
    long["Px"] = long["Px"].ffill()

    long = long[[c for c in FINAL_COLS if c in long.columns]]
    for c in INT_COLS:
        if c in long.columns:
            long[c] = pd.to_numeric(long[c], errors="coerce").astype("Int64")
    long["Px"] = pd.to_numeric(long["Px"], errors="coerce").astype("Float64")
    return long


def upsert(new_data, fetch_start):
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        existing["Date"] = pd.to_datetime(existing["Date"])
        mask = ~(
            existing["Commodity"].isin(new_data["Commodity"].unique()) &
            (existing["Date"] >= pd.Timestamp(fetch_start))
        )
        final = pd.concat([existing[mask], new_data], ignore_index=True)
    else:
        final = new_data
    final = final.sort_values(["Commodity", "Crop", "Date"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PATH, index=False)
    return final


def incremental_start():
    existing = pd.read_parquet(OUT_PATH, columns=["Date"])
    existing["Date"] = pd.to_datetime(existing["Date"])
    latest = existing["Date"].max()
    return (latest - pd.Timedelta(days=30)).strftime("%Y-%m-%d")


# ── MAIN ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true")
args = parser.parse_args()

END_DATE = datetime.today().strftime("%Y-%m-%d")

if not args.full and OUT_PATH.exists():
    fetch_start = incremental_start()
    print(f"INCREMENTAL from {fetch_start}\n")
else:
    fetch_start = START_DATE
    print(f"FULL rebuild from {fetch_start}\n")

all_dfs = []
for comm, cfg in COMMODITIES.items():
    print(f"-- {comm} --")
    df = build_commodity(comm, cfg, fetch_start, END_DATE)
    if df is not None:
        all_dfs.append(df)

if all_dfs:
    new_data = pd.concat(all_dfs, ignore_index=True)
    final = upsert(new_data, fetch_start)
    print(f"\nSaved -> {OUT_PATH}  |  {len(final)} rows")
    print(final.groupby(["Commodity", "Crop"]).agg(
        rows=("Date", "count"),
        from_=("Date", "min"),
        to=("Date", "max")
    ).to_string())
else:
    print("No data retrieved.")
