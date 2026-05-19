"""
cot_backfill.py — Full COT backfill from ICE Connect
=====================================================
Produces three parquets in COT_ALL/Database:

  cot_cit.parquet
    Commodity, Date,
    Comm Long/Short, Spec Long/Short/Spread, Index Long/Short,
    Non Rep Long/Short, Total OI,
    Traders [Comm/Spec/Index/Tot Rept],
    Pct OI [all position groups],
    Px

  cot_disagg_futopt.parquet
    Commodity, Crop (All/Old/Other), Date,
    Producer Long/Short, Swap Long/Short/Spread,
    MM Long/Short/Spread, Other Long/Short/Spread,
    Tot Rept Long/Short, Non Rep Long/Short, Total OI,
    Traders [all groups + Total + Tot Rept],
    Conc Gross/Net 4/8 Long/Short,
    Pct OI [all position groups],
    Px

  cot_disagg_fut.parquet
    [identical schema to cot_disagg_futopt.parquet]

  Options = cot_disagg_futopt - cot_disagg_fut  (derived at query time)

Usage:
    python cot_backfill.py                          # incremental all
    python cot_backfill.py --full                   # full rebuild
    python cot_backfill.py --full --commodity KC    # single commodity
    python cot_backfill.py --cit                    # CIT only
    python cot_backfill.py --disagg                 # Disagg only
    python cot_backfill.py --full --start 2026-03-01 --commodity KC  # KC 10-week test
"""

import argparse
import json
import sys
import threading
import icepython as ice
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

CODE_DIR      = Path(__file__).parent
DB_DIR        = CODE_DIR.parent / "Database"
AUTOMATOR_DIR = CODE_DIR.parent / "Automator"
CIT_PATH         = DB_DIR / "cot_cit.parquet"
DISAGG_FUTOPT_PATH = DB_DIR / "cot_disagg_futopt.parquet"
DISAGG_FUT_PATH    = DB_DIR / "cot_disagg_fut.parquet"

CIT_START    = "2010-01-01"
DISAGG_START = "2010-01-01"

# ── Commodity configs ─────────────────────────────────────────────────────────
CIT_COMMODITIES = {
    "KC": {"cot_sym": "KC #COMB-CFTC", "px_sym": "%KC 1!"},
    "CC": {"cot_sym": "CC #COMB-CFTC", "px_sym": "%CC 1!"},
    "SB": {"cot_sym": "SB #COMB-CFTC", "px_sym": "%SB 1!"},
    "CT": {"cot_sym": "CT #COMB-CFTC", "px_sym": "%CT 1!"},
}

DISAGG_COMMODITIES = {
    "KC":  {"comb_sym": "KC #COMB-CFTC",     "fut_sym": "KC #FUT-CFTC",     "px_sym": "%KC 1!"},
    "CC":  {"comb_sym": "CC #COMB-CFTC",     "fut_sym": "CC #FUT-CFTC",     "px_sym": "%CC 1!"},
    "SB":  {"comb_sym": "SB #COMB-CFTC",     "fut_sym": "SB #FUT-CFTC",     "px_sym": "%SB 1!"},
    "CT":  {"comb_sym": "CT #COMB-CFTC",     "fut_sym": "CT #FUT-CFTC",     "px_sym": "%CT 1!"},
    "RC":  {"comb_sym": "RC.ICE #COMB-CFTC", "fut_sym": "RC.ICE #FUT-CFTC", "px_sym": "%RC 1!-ICE"},
    "LCC": {"comb_sym": "C.ICE #COMB-CFTC",  "fut_sym": "C.ICE #FUT-CFTC",  "px_sym": "%C 1!-ICE"},
    "LSU": {"comb_sym": "W.ICE #COMB-CFTC",  "fut_sym": "W.ICE #FUT-CFTC",  "px_sym": "%W 1!-ICE"},
}

# ── CIT fields ────────────────────────────────────────────────────────────────
CIT_POS_FIELDS = [
    "Open Interest All Close",
    "Comm Positions Long All Nocit Close",
    "Comm Positions Short All Nocit Close",
    "Ncomm Positions Long All Nocit Close",
    "Ncomm Positions Short All Nocit Close",
    "Ncomm Positions Spread All Nocit Close",
    "Cit Positions Long All Close",
    "Cit Positions Short All Close",
]
CIT_POS_CLEAN = [
    "Total OI",
    "Comm Long", "Comm Short",
    "Spec Long", "Spec Short", "Spec Spread",
    "Index Long", "Index Short",
]

CIT_TRADER_FIELDS = [
    "Traders Comm Long All Nocit Close",
    "Traders Comm Short All Nocit Close",
    "Traders Noncomm Long All Nocit Close",
    "Traders Noncomm Short All Nocit Close",
    "Traders Noncomm Spread All Nocit Close",
    "Traders Cit Long All Close",
    "Traders Cit Short All Close",
    "Traders Tot Rept Long All Nocit Close",
    "Traders Tot Rept Short All Nocit Close",
]
CIT_TRADER_CLEAN = [
    "Traders Comm Long", "Traders Comm Short",
    "Traders Spec Long", "Traders Spec Short", "Traders Spec Spread",
    "Traders Index Long", "Traders Index Short",
    "Traders Tot Rept Long", "Traders Tot Rept Short",
]

CIT_ALL_FIELDS = CIT_POS_FIELDS + CIT_TRADER_FIELDS
CIT_ALL_CLEAN  = CIT_POS_CLEAN  + CIT_TRADER_CLEAN

CIT_POS_FOR_PCT = [
    "Comm Long", "Comm Short",
    "Spec Long", "Spec Short", "Spec Spread",
    "Index Long", "Index Short",
    "Non Rep Long", "Non Rep Short",
]

CIT_INT_COLS = CIT_POS_CLEAN + ["Non Rep Long", "Non Rep Short"] + CIT_TRADER_CLEAN

CIT_FINAL_COLS = (
    ["Commodity", "Date"]
    + ["Comm Long", "Comm Short",
       "Spec Long", "Spec Short", "Spec Spread",
       "Index Long", "Index Short",
       "Non Rep Long", "Non Rep Short", "Total OI"]
    + CIT_TRADER_CLEAN
    + ["Pct OI " + c for c in CIT_POS_FOR_PCT]
    + ["Px"]
)

# ── Disagg fields (crop-parameterised) ───────────────────────────────────────
def _pos_fields(crop):
    return [
        f"Open Interest {crop} Close",
        f"Prod Merc Positions Long {crop} Close",
        f"Prod Merc Positions Short {crop} Close",
        f"Swap Positions Long {crop} Close",
        f"Swap Positions Short {crop} Close",
        f"Swap Positions Spread {crop} Close",
        f"M Money Positions Long {crop} Close",
        f"M Money Positions Short {crop} Close",
        f"M Money Positions Spread {crop} Close",
        f"Other Rept Positions Long {crop} Close",
        f"Other Rept Positions Short {crop} Close",
        f"Other Rept Positions Spread {crop} Close",
        f"Tot Rept Positions Long {crop} Close",
        f"Tot Rept Positions Short {crop} Close",
        f"Nonrept Positions Long {crop} Close",
        f"Nonrept Positions Short {crop} Close",
    ]

def _trader_fields(crop):
    return [
        f"Traders Tot {crop} Close",
        f"Traders Prod Merc Long {crop} Close",
        f"Traders Prod Merc Short {crop} Close",
        f"Traders Swap Long {crop} Close",
        f"Traders Swap Short {crop} Close",
        f"Traders Swap Spread {crop} Close",
        f"Traders M Money Long {crop} Close",
        f"Traders M Money Short {crop} Close",
        f"Traders M Money Spread {crop} Close",
        f"Traders Other Rept Long {crop} Close",
        f"Traders Other Rept Short {crop} Close",
        f"Traders Other Rept Spread {crop} Close",
        f"Traders Tot Rept Long {crop} Close",
        f"Traders Tot Rept Short {crop} Close",
    ]

def _conc_fields(crop):
    return [
        f"Conc Gross Le 4 Tdr Long {crop} Close",
        f"Conc Gross Le 4 Tdr Short {crop} Close",
        f"Conc Gross Le 8 Tdr Long {crop} Close",
        f"Conc Gross Le 8 Tdr Short {crop} Close",
        f"Conc Net Le 4 Tdr Long {crop} Close",
        f"Conc Net Le 4 Tdr Short {crop} Close",
        f"Conc Net Le 8 Tdr Long {crop} Close",
        f"Conc Net Le 8 Tdr Short {crop} Close",
    ]

DISAGG_POS_CLEAN = [
    "Total OI",
    "Producer Long", "Producer Short",
    "Swap Long", "Swap Short", "Swap Spread",
    "MM Long", "MM Short", "MM Spread",
    "Other Long", "Other Short", "Other Spread",
    "Tot Rept Long", "Tot Rept Short",
    "Non Rep Long", "Non Rep Short",
]
DISAGG_TRADER_CLEAN = [
    "Traders Total",
    "Traders Producer Long", "Traders Producer Short",
    "Traders Swap Long", "Traders Swap Short", "Traders Swap Spread",
    "Traders MM Long", "Traders MM Short", "Traders MM Spread",
    "Traders Other Long", "Traders Other Short", "Traders Other Spread",
    "Traders Tot Rept Long", "Traders Tot Rept Short",
]
DISAGG_CONC_CLEAN = [
    "Conc Gross 4 Long", "Conc Gross 4 Short",
    "Conc Gross 8 Long", "Conc Gross 8 Short",
    "Conc Net 4 Long",   "Conc Net 4 Short",
    "Conc Net 8 Long",   "Conc Net 8 Short",
]
DISAGG_ALL_CLEAN = DISAGG_POS_CLEAN + DISAGG_TRADER_CLEAN + DISAGG_CONC_CLEAN

DISAGG_POS_FOR_PCT = [c for c in DISAGG_POS_CLEAN if c != "Total OI"]

DISAGG_INT_COLS   = DISAGG_POS_CLEAN + DISAGG_TRADER_CLEAN
DISAGG_FLOAT_COLS = DISAGG_CONC_CLEAN + ["Pct OI " + c for c in DISAGG_POS_FOR_PCT]

DISAGG_FINAL_COLS = (
    ["Commodity", "Crop", "Date"]
    + DISAGG_POS_CLEAN
    + DISAGG_TRADER_CLEAN
    + DISAGG_CONC_CLEAN
    + ["Pct OI " + c for c in DISAGG_POS_FOR_PCT]
    + ["Px"]
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_ts(symbol, fields, start, end):
    def _call():
        data = ice.get_timeseries(symbol, fields, granularity="D",
                                  start_date=start, end_date=end)
        return list(data)
    for attempt in range(FETCH_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                rows = ex.submit(_call).result(timeout=FETCH_TIMEOUT)
            if not rows or (rows[0] and "Error" in str(rows[0][0])):
                return None
            df = pd.DataFrame(rows[1:], columns=["Date"] + fields)
            df["Date"] = pd.to_datetime(df["Date"])
            for f in fields:
                df[f] = pd.to_numeric(df[f], errors="coerce")
            return df.set_index("Date")
        except TimeoutError:
            if attempt < FETCH_RETRIES:
                print(f"  TIMEOUT {symbol} — retry {attempt + 1}/{FETCH_RETRIES}...")
            else:
                print(f"  TIMEOUT {symbol} (>{FETCH_TIMEOUT}s x{FETCH_RETRIES + 1} attempts)")
                with _failures_lock:
                    FETCH_FAILURES.append(symbol)
                return None
        except Exception as e:
            print(f"  ERROR {symbol}: {e}")
            with _failures_lock:
                FETCH_FAILURES.append(symbol)
            return None
    return None


def fetch_ts_chunked(symbol, fields, start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    chunks, year = [], s.year
    while True:
        cs = max(s, pd.Timestamp(f"{year}-01-01"))
        ce = min(e, pd.Timestamp(f"{year}-12-31"))
        df = fetch_ts(symbol, fields, cs.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            chunks.append(df)
        year += 1
        if ce >= e:
            break
    return pd.concat(chunks).sort_index() if chunks else None


def fetch_px(symbol, start, end):
    def _call():
        data = ice.get_timeseries(symbol, ["Settle"], granularity="D",
                                  start_date=start, end_date=end)
        return list(data)
    for attempt in range(FETCH_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                rows = ex.submit(_call).result(timeout=FETCH_TIMEOUT)
            if not rows or (rows[0] and "Error" in str(rows[0][0])):
                return pd.Series(dtype=float)
            df = pd.DataFrame(rows[1:], columns=["Date", "Px"])
            df["Date"] = pd.to_datetime(df["Date"])
            df["Px"] = pd.to_numeric(df["Px"], errors="coerce")
            return df.set_index("Date")["Px"]
        except TimeoutError:
            if attempt < FETCH_RETRIES:
                print(f"  TIMEOUT px {symbol} — retry {attempt + 1}/{FETCH_RETRIES}...")
            else:
                print(f"  TIMEOUT px {symbol} (>{FETCH_TIMEOUT}s x{FETCH_RETRIES + 1} attempts)")
                with _failures_lock:
                    FETCH_FAILURES.append(f"px:{symbol}")
                return pd.Series(dtype=float)
        except Exception as e:
            print(f"  ERROR px {symbol}: {e}")
            with _failures_lock:
                FETCH_FAILURES.append(f"px:{symbol}")
            return pd.Series(dtype=float)
    return pd.Series(dtype=float)


def upsert(db_path, new_data, fetch_start, key_cols):
    if db_path.exists():
        existing = pd.read_parquet(db_path)
        existing["Date"] = pd.to_datetime(existing["Date"])
        mask = pd.Series([True] * len(existing), index=existing.index)
        for col in key_cols:
            mask &= existing[col].isin(new_data[col].unique())
        mask &= existing["Date"] >= pd.Timestamp(fetch_start)
        final = pd.concat([existing[~mask], new_data], ignore_index=True)
    else:
        final = new_data
    final = final.sort_values(key_cols + ["Date"]).reset_index(drop=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(db_path, index=False)
    return final


def incremental_start(db_path):
    existing = pd.read_parquet(db_path, columns=["Date"])
    latest = pd.to_datetime(existing["Date"]).max()
    return (latest - pd.Timedelta(days=14)).strftime("%Y-%m-%d")


def add_pct_oi(df, pos_cols):
    for c in pos_cols:
        if c in df.columns and "Total OI" in df.columns:
            df[f"Pct OI {c}"] = (df[c] / df["Total OI"] * 100).round(2)
    return df


# ── CIT builder ───────────────────────────────────────────────────────────────
def build_cit(comm, cfg, start, end):
    sym = cfg["cot_sym"]
    print(f"  [{comm}] CIT  {sym}...", end=" ", flush=True)

    df = fetch_ts(sym, CIT_ALL_FIELDS, start, end)
    if df is None or df.empty:
        print("no data"); return None
    print(f"{len(df)} rows")

    if len(df.columns) != len(CIT_ALL_CLEAN):
        print(f"  [{comm}] CIT column count mismatch: got {len(df.columns)}, expected {len(CIT_ALL_CLEAN)}"); return None
    df.columns = CIT_ALL_CLEAN
    df["Non Rep Long"]  = df["Total OI"] - df["Comm Long"]  - df["Spec Long"]  - df["Spec Spread"] - df["Index Long"]
    df["Non Rep Short"] = df["Total OI"] - df["Comm Short"] - df["Spec Short"] - df["Spec Spread"] - df["Index Short"]
    df = add_pct_oi(df, CIT_POS_FOR_PCT)

    print(f"    Px {cfg['px_sym']}...", end=" ", flush=True)
    px = fetch_px(cfg["px_sym"], start, end)
    print(f"{len(px)} rows")
    df["Px"] = px.reindex(px.index.union(df.index)).ffill().reindex(df.index)

    df["Commodity"] = comm
    df = df.reset_index()
    df = df[[c for c in CIT_FINAL_COLS if c in df.columns]]

    for c in CIT_INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in [col for col in df.columns if col.startswith("Pct OI")] + ["Px"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
    return df


# ── Disagg builder ────────────────────────────────────────────────────────────
def _fetch_disagg_slice(symbol, crop, start, end):
    all_fields = _pos_fields(crop) + _trader_fields(crop) + _conc_fields(crop)
    chunked = (crop != "All")
    fn = fetch_ts_chunked if chunked else fetch_ts
    df = fn(symbol, all_fields, start, end)
    if df is None or df.empty:
        return None
    if len(df.columns) != len(DISAGG_ALL_CLEAN):
        print(f"  column count mismatch ({crop}): got {len(df.columns)}, expected {len(DISAGG_ALL_CLEAN)}"); return None
    df.columns = DISAGG_ALL_CLEAN
    df["Crop"] = crop
    return df


def build_disagg(comm, cfg, version, start, end):
    symbol = cfg["comb_sym"] if version == "FutOpt" else cfg["fut_sym"]
    pieces = []

    with ThreadPoolExecutor(max_workers=3) as inner:
        crop_futures = {inner.submit(_fetch_disagg_slice, symbol, crop, start, end): crop
                        for crop in ("All", "Old", "Other")}
        for f in as_completed(crop_futures):
            crop = crop_futures[f]
            df = f.result()
            print(f"  [{comm}] Disagg {version} {crop}  {symbol}... ", end="", flush=True)
            if df is None:
                print("no data"); continue
            print(f"{len(df)} rows")
            pieces.append(df)

    if not pieces:
        return None

    combined = pd.concat(pieces).reset_index()

    print(f"    Px {cfg['px_sym']}...", end=" ", flush=True)
    px = fetch_px(cfg["px_sym"], start, end)
    print(f"{len(px)} rows")
    px_df = px.reset_index(); px_df.columns = ["Date", "Px"]
    combined = combined.merge(px_df, on="Date", how="left")
    combined = combined.sort_values(["Crop", "Date"])
    combined["Px"] = combined.groupby("Crop")["Px"].ffill()

    combined = add_pct_oi(combined, DISAGG_POS_FOR_PCT)
    combined["Commodity"] = comm
    combined = combined[[c for c in DISAGG_FINAL_COLS if c in combined.columns]]

    for c in DISAGG_INT_COLS:
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c], errors="coerce").astype("Int64")
    for c in DISAGG_FLOAT_COLS + ["Px"]:
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c], errors="coerce").astype("Float64")
    return combined


FETCH_TIMEOUT  = 60   # seconds per ICE request before giving up
FETCH_RETRIES  = 1    # retry once on timeout before marking as failed
_failures_lock = threading.Lock()
FETCH_FAILURES: list[str] = []

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",      action="store_true")
    parser.add_argument("--cit",       action="store_true")
    parser.add_argument("--disagg",    action="store_true")
    parser.add_argument("--commodity", type=str, default=None)
    parser.add_argument("--start",     type=str, default=None)
    args = parser.parse_args()

    run_cit    = not args.disagg
    run_disagg = not args.cit
    END_DATE   = datetime.today().strftime("%Y-%m-%d")

    if args.commodity:
        comm = args.commodity.upper()
        if comm not in CIT_COMMODITIES and comm not in DISAGG_COMMODITIES:
            print(f"Unknown commodity: {comm}"); sys.exit(1)
        CIT_COMMODITIES    = {comm: CIT_COMMODITIES[comm]}    if comm in CIT_COMMODITIES    else {}
        DISAGG_COMMODITIES = {comm: DISAGG_COMMODITIES[comm]} if comm in DISAGG_COMMODITIES else {}
        if not CIT_COMMODITIES:    run_cit    = False
        if not DISAGG_COMMODITIES: run_disagg = False


    def _get_start(path, key_cols):
        if args.start:
            return args.start, f"FULL from {args.start} (--start override)"
        if args.full or not path.exists():
            default = CIT_START if "cit" in path.name else DISAGG_START
            return default, f"FULL from {default}"
        s = incremental_start(path)
        return s, f"INCREMENTAL from {s}"


    MAX_WORKERS = 6

    # ── CIT ───────────────────────────────────────────────────────────────────────
    if run_cit:
        fetch_start, mode = _get_start(CIT_PATH, ["Commodity"])
        print(f"\nCIT | {mode}\n")

        all_cit = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(build_cit, comm, cfg, fetch_start, END_DATE): comm
                       for comm, cfg in CIT_COMMODITIES.items()}
            for f in as_completed(futures):
                df = f.result()
                if df is not None:
                    all_cit.append(df)

        if all_cit:
            final = upsert(CIT_PATH, pd.concat(all_cit, ignore_index=True),
                           fetch_start, ["Commodity"])
            print(f"\nCIT saved -> {CIT_PATH}  |  {len(final)} rows")
            print(final.groupby("Commodity").agg(
                rows=("Date", "count"), from_=("Date", "min"), to=("Date", "max")).to_string())
        else:
            print("CIT: no data.")

    # ── Disagg FutOpt + Fut (combined parallel phase) ─────────────────────────────
    if run_disagg:
        fo_start,  fo_mode  = _get_start(DISAGG_FUTOPT_PATH, ["Commodity", "Crop"])
        fut_start, fut_mode = _get_start(DISAGG_FUT_PATH,    ["Commodity", "Crop"])
        print(f"\nDISAGG FutOpt | {fo_mode}")
        print(f"DISAGG Fut    | {fut_mode}\n")

        all_fo, all_fut = [], []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for comm, cfg in DISAGG_COMMODITIES.items():
                futures[pool.submit(build_disagg, comm, cfg, "FutOpt", fo_start,  END_DATE)] = ("FutOpt", fo_start)
                futures[pool.submit(build_disagg, comm, cfg, "Fut",    fut_start, END_DATE)] = ("Fut",    fut_start)
            for f in as_completed(futures):
                version, _ = futures[f]
                df = f.result()
                if df is not None:
                    (all_fo if version == "FutOpt" else all_fut).append(df)

        if all_fo:
            final = upsert(DISAGG_FUTOPT_PATH, pd.concat(all_fo, ignore_index=True),
                           fo_start, ["Commodity", "Crop"])
            print(f"\nDisagg FutOpt saved -> {DISAGG_FUTOPT_PATH}  |  {len(final)} rows")
            print(final.groupby(["Commodity", "Crop"]).agg(
                rows=("Date", "count"), from_=("Date", "min"), to=("Date", "max")).to_string())
        else:
            print("Disagg FutOpt: no data.")

        if all_fut:
            final = upsert(DISAGG_FUT_PATH, pd.concat(all_fut, ignore_index=True),
                           fut_start, ["Commodity", "Crop"])
            print(f"\nDisagg Fut saved -> {DISAGG_FUT_PATH}  |  {len(final)} rows")
            print(final.groupby(["Commodity", "Crop"]).agg(
                rows=("Date", "count"), from_=("Date", "min"), to=("Date", "max")).to_string())
        else:
            print("Disagg Fut: no data.")

    # ── Write failure manifest ─────────────────────────────────────────────────────
    failures_path = AUTOMATOR_DIR / "failures.json"
    unique_failures = sorted(set(FETCH_FAILURES))
    with open(failures_path, "w") as fh:
        json.dump({"run_time": END_DATE, "failed_symbols": unique_failures}, fh, indent=2)
    if unique_failures:
        print(f"\nWARNING: {len(unique_failures)} symbol(s) failed — {unique_failures}")
        print(f"Details written to {failures_path}")
