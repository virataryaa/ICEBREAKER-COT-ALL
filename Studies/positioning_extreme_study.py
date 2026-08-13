"""
positioning_extreme_study.py — standalone backtest of the Positioning
Extreme contrarian-reversal hypothesis, across ALL seven commodities
(not just the four CIT ones tested in cot_positioning_study.py's Study 1).

Hypothesis: when the pure-speculative category's gross Long (or Short) is
at a point-in-time historical high, that crowding is unsustainable and
price tends to reverse over the following weeks — the classic "fade the
crowd" contrarian trade.

Coverage:
  KC, CC, SB, CT  — CIT report, Spec Long/Short (pure large-speculator leg)
  RC, LCC, LSU    — Disaggregated report, MM Long/Short (Managed Money is
                     the Disagg report's equivalent pure-speculator leg)

Both categories are the same economic thing (large discretionary
speculators, no hedging/spreading contamination) under two different CFTC
report formats, so pooling all seven is apples-to-apples.

Run:  python positioning_extreme_study.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DB_DIR      = Path(__file__).resolve().parent.parent / "Database"
CIT_FILE    = DB_DIR / "cot_cit.parquet"
DISAGG_FILE = DB_DIR / "cot_disagg_fut.parquet"
ROLLEX_DIR  = DB_DIR / "Rollex"
ROLLEX_MAP  = {"KC": "rollex_KC.parquet", "CC": "rollex_CC.parquet",
               "SB": "rollex_SB.parquet", "CT": "rollex_CT.parquet",
               "RC": "rollex_RC.parquet", "LCC": "rollex_LCC.parquet",
               "LSU": "rollex_LSU.parquet"}
CIT_SYMS    = {"KC", "CC", "SB", "CT"}
DISAGG_SYMS = {"RC", "LCC", "LSU"}

MIN_HISTORY_WEEKS = 104
N_BOOT   = 5000
BLOCK    = 8
COST_BPS = 4.0
HOLD     = 4
RNG      = np.random.default_rng(13)


# ══════════════════════════════════════════════════════════════════════════
# Data loading — normalises CIT's "Spec Long/Short" and Disagg's
# "MM Long/Short" into a common SpecLong/SpecShort pair (both are the
# report's pure large-speculator category, just named differently).
# ══════════════════════════════════════════════════════════════════════════
def load_commodity(sym: str) -> pd.DataFrame:
    rx = pd.read_parquet(ROLLEX_DIR / ROLLEX_MAP[sym], columns=["rollex_px"])
    rx.index = pd.to_datetime(rx.index)
    rx.index.name = "Date"
    rx = rx.reset_index().sort_values("Date")

    if sym in CIT_SYMS:
        cot = pd.read_parquet(CIT_FILE)
        cot = cot[cot["Commodity"] == sym].sort_values("Date").reset_index(drop=True)
        cot["Date"] = pd.to_datetime(cot["Date"])
        cot = cot.rename(columns={"Spec Long": "SpecLong", "Spec Short": "SpecShort"})
    else:
        cot = pd.read_parquet(DISAGG_FILE)
        cot = cot[(cot["Commodity"] == sym) & (cot["Crop"] == "All")].sort_values("Date").reset_index(drop=True)
        cot["Date"] = pd.to_datetime(cot["Date"])
        cot = cot.rename(columns={"MM Long": "SpecLong", "MM Short": "SpecShort"})

    d = pd.merge_asof(cot[["Date", "SpecLong", "SpecShort"]], rx, on="Date", direction="backward")
    d = d.dropna(subset=["rollex_px", "SpecLong", "SpecShort"]).reset_index(drop=True)
    d["category"] = "Spec" if sym in CIT_SYMS else "MM"
    return d


def expanding_percentile(s: pd.Series, min_periods=MIN_HISTORY_WEEKS) -> np.ndarray:
    vals = s.values.astype(float)
    out = np.full(len(vals), np.nan)
    for i in range(min_periods - 1, len(vals)):
        window = vals[: i + 1]
        out[i] = (window <= vals[i]).mean() * 100.0
    return out


def forward_return(px: pd.Series, n: int) -> np.ndarray:
    return (px.shift(-n).values / px.values - 1.0) * 100.0


def block_bootstrap_pvalue(mask: np.ndarray, y: np.ndarray, block=BLOCK, n_boot=N_BOOT) -> tuple:
    valid = ~np.isnan(y) & ~np.isnan(mask.astype(float))
    y, mask = y[valid], mask[valid]
    if mask.sum() < 8 or (~mask).sum() < 8:
        return np.nan, np.nan, np.nan
    obs_diff = y[mask].mean() - y[~mask].mean()
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        ys, ms = y[idx], mask[idx]
        if ms.sum() == 0 or (~ms).sum() == 0:
            diffs[b] = np.nan
            continue
        diffs[b] = ys[ms].mean() - ys[~ms].mean()
    diffs = diffs[~np.isnan(diffs)]
    p = float((np.abs(diffs) >= abs(obs_diff)).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs_diff, p, (lo, hi)


def pooled_block_bootstrap_pvalue(groups: list, block=BLOCK, n_boot=N_BOOT) -> tuple:
    clean = []
    for mask, y in groups:
        valid = ~np.isnan(y) & ~np.isnan(mask.astype(float))
        m, yv = mask[valid], y[valid]
        if len(yv) > 0:
            clean.append((m, yv))
    if not clean:
        return np.nan, np.nan, np.nan, 0
    all_mask = np.concatenate([m for m, _ in clean])
    all_y = np.concatenate([y for _, y in clean])
    if all_mask.sum() < 8 or (~all_mask).sum() < 8:
        return np.nan, np.nan, np.nan, int(all_mask.sum())
    obs_diff = all_y[all_mask].mean() - all_y[~all_mask].mean()
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        bm_parts, by_parts = [], []
        for m, y in clean:
            n = len(y)
            if n <= block:
                bm_parts.append(m); by_parts.append(y); continue
            n_blocks = int(np.ceil(n / block))
            starts = RNG.integers(0, n - block, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            bm_parts.append(m[idx]); by_parts.append(y[idx])
        bm, by = np.concatenate(bm_parts), np.concatenate(by_parts)
        if bm.sum() == 0 or (~bm).sum() == 0:
            diffs[b] = np.nan
            continue
        diffs[b] = by[bm].mean() - by[~bm].mean()
    diffs = diffs[~np.isnan(diffs)]
    p = float((np.abs(diffs) >= abs(obs_diff)).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs_diff, p, (lo, hi), int(all_mask.sum())


def bh_correct(pvals: list) -> list:
    pvals = np.array(pvals, dtype=float)
    n = np.sum(~np.isnan(pvals))
    order = np.argsort(np.where(np.isnan(pvals), np.inf, pvals))
    ranks = np.empty(len(pvals)); ranks[order] = np.arange(1, len(pvals) + 1)
    adj = pvals * n / np.where(ranks == 0, 1, ranks)
    return list(np.minimum.accumulate(adj[order][::-1])[::-1][np.argsort(order)])


def backtest_stats(bt: pd.DataFrame, split_frac=0.7) -> dict:
    n = len(bt)
    cut = int(n * split_frac)
    out = {}
    for label, sub in [("full", bt), ("in-sample (first 70%)", bt.iloc[:cut]),
                        ("out-of-sample (last 30%)", bt.iloc[cut:])]:
        r = sub["ret"].values
        std = r.std(ddof=1)
        sharpe = (r.mean() / std * np.sqrt(52)) if std > 0 else np.nan
        equity = (1 + r).cumprod()
        dd = (equity / np.maximum.accumulate(equity) - 1).min() if len(equity) else np.nan
        bh_equity = (1 + sub["bh_ret"].values).cumprod()
        out[label] = dict(weeks=len(sub), total_ret_pct=(equity[-1] - 1) * 100 if len(equity) else np.nan,
                           bh_total_ret_pct=(bh_equity[-1] - 1) * 100 if len(bh_equity) else np.nan,
                           ann_sharpe=sharpe, max_dd_pct=dd * 100 if pd.notna(dd) else np.nan,
                           pct_weeks_in_market=(sub["pos"] != 0).mean() * 100)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Positioning Extreme — contrarian reversal
# ══════════════════════════════════════════════════════════════════════════
def extreme_masks(d: pd.DataFrame, thresh: float):
    pctL = expanding_percentile(d["SpecLong"])
    pctS = expanding_percentile(d["SpecShort"])
    return (pctL >= thresh), (pctS >= thresh)


def bucket_tests(data_by_sym: dict, thresh: float, horizons=(1, 2, 4, 8)) -> list:
    rows = []
    for sym, d in data_by_sym.items():
        crowded_long, crowded_short = extreme_masks(d, thresh)
        for h in horizons:
            fwd = forward_return(d["rollex_px"], h)
            diff, p, ci = block_bootstrap_pvalue(crowded_long, fwd)
            rows.append(dict(symbol=sym, side="crowded_long", horizon=h, thresh=thresh,
                              n=int(np.nansum(crowded_long)), mean_diff_pct=diff, p=p))
            diff, p, ci = block_bootstrap_pvalue(crowded_short, fwd)
            rows.append(dict(symbol=sym, side="crowded_short", horizon=h, thresh=thresh,
                              n=int(np.nansum(crowded_short)), mean_diff_pct=diff, p=p))
    return rows


def pooled_bucket_tests(data_by_sym: dict, thresh: float, horizons=(1, 2, 4, 8)) -> list:
    rows = []
    masks = {sym: extreme_masks(d, thresh) for sym, d in data_by_sym.items()}
    for h in horizons:
        groups_long  = [(masks[sym][0], forward_return(d["rollex_px"], h)) for sym, d in data_by_sym.items()]
        diff, p, ci, n = pooled_block_bootstrap_pvalue(groups_long)
        rows.append(dict(symbol="POOLED(7)", side="crowded_long", horizon=h, thresh=thresh,
                          n=n, mean_diff_pct=diff, p=p))
        groups_short = [(masks[sym][1], forward_return(d["rollex_px"], h)) for sym, d in data_by_sym.items()]
        diff, p, ci, n = pooled_block_bootstrap_pvalue(groups_short)
        rows.append(dict(symbol="POOLED(7)", side="crowded_short", horizon=h, thresh=thresh,
                          n=n, mean_diff_pct=diff, p=p))
    return rows


def contrarian_backtest(d: pd.DataFrame, thresh=90, hold=HOLD) -> pd.DataFrame:
    """Short when crowded long, long when crowded short, `hold` weeks each,
    lagged 1 week entry (signal known when COT prints), no pyramiding."""
    crowded_long, crowded_short = extreme_masks(d, thresh)
    n = len(d)
    pos = np.zeros(n)
    for i in np.where(crowded_long)[0]:
        entry = i + 1
        if entry < n:
            pos[entry: min(entry + hold, n)] = -1.0
    for i in np.where(crowded_short)[0]:
        entry = i + 1
        if entry < n:
            pos[entry: min(entry + hold, n)] = 1.0   # crowded-short signal wins ties (rare overlap)

    px = d["rollex_px"].values
    ret_1w = np.nan_to_num(np.r_[np.nan, px[1:] / px[:-1] - 1.0])
    strat_ret = pos * ret_1w
    cost = np.r_[0, np.abs(np.diff(pos))] * (COST_BPS / 10000.0)
    return pd.DataFrame(dict(Date=d["Date"].values, ret=strat_ret - cost, bh_ret=ret_1w, pos=pos))


def pooled_contrarian_backtest(data_by_sym: dict, thresh=90, hold=HOLD) -> pd.DataFrame:
    per_sym = {}
    for sym, d in data_by_sym.items():
        bt = contrarian_backtest(d, thresh, hold)
        per_sym[sym] = bt.rename(columns={"ret": f"ret_{sym}", "bh_ret": f"bh_{sym}", "pos": f"pos_{sym}"})
    merged = None
    for bt in per_sym.values():
        merged = bt if merged is None else merged.merge(bt, on="Date", how="inner")
    ret_cols = [c for c in merged.columns if c.startswith("ret_")]
    bh_cols  = [c for c in merged.columns if c.startswith("bh_")]
    pos_cols = [c for c in merged.columns if c.startswith("pos_")]
    return pd.DataFrame(dict(Date=merged["Date"], ret=merged[ret_cols].mean(axis=1),
                              bh_ret=merged[bh_cols].mean(axis=1),
                              pos=merged[pos_cols].abs().sum(axis=1).clip(upper=1)))


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════
def main():
    data_by_sym = {}
    for sym in ["KC", "CC", "SB", "CT", "RC", "LCC", "LSU"]:
        d = load_commodity(sym)
        data_by_sym[sym] = d
        cat = d["category"].iloc[0]
        print(f"{sym:5s} ({cat:4s})  {len(d):4d} weeks  {d['Date'].min().date()} to {d['Date'].max().date()}")

    all_rows = []
    for thresh in (80, 85, 90):
        all_rows += bucket_tests(data_by_sym, thresh)
        all_rows += pooled_bucket_tests(data_by_sym, thresh)

    df = pd.DataFrame(all_rows)
    df["p_bh"] = bh_correct(df["p"].tolist())

    print(f"\n{'='*70}\nBUCKET TESTS: crowded-long / crowded-short vs forward return, per symbol "
          f"and pooled across all 7\n{'='*70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df[["symbol", "side", "thresh", "horizon", "n", "mean_diff_pct", "p", "p_bh"]]
              .round(3).to_string(index=False))

    sig = df[df["p_bh"] <= 0.10]
    print(f"\n{'='*70}\nSURVIVES q<=0.10 after FDR correction: {len(sig)} / {len(df)}\n{'='*70}")
    if len(sig):
        print(sig[["symbol", "side", "thresh", "horizon", "mean_diff_pct", "p_bh"]].to_string(index=False))

    # ── Backtest: strict (thresh=90) per commodity + pooled ─────────────
    print(f"\n\n{'='*70}\nCONTRARIAN BACKTEST (short crowded-long / long crowded-short, "
          f"thresh=90, {HOLD}wk hold, {COST_BPS}bps cost, signal lagged 1wk)\n{'='*70}")
    for sym, d in data_by_sym.items():
        stats = backtest_stats(contrarian_backtest(d, thresh=90))
        print(f"\n  {sym}:")
        for label, s in stats.items():
            print(f"    {label:28s}  weeks={s['weeks']:4d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
                  f"b&h_ret={s['bh_total_ret_pct']:+7.1f}%  ann_sharpe={s['ann_sharpe']:+.2f}  "
                  f"max_dd={s['max_dd_pct']:6.1f}%  in_mkt={s['pct_weeks_in_market']:5.1f}%")

    pooled_bt = pooled_contrarian_backtest(data_by_sym, thresh=90)
    pooled_stats = backtest_stats(pooled_bt)
    print(f"\n  POOLED (equal-weight all 7):")
    for label, s in pooled_stats.items():
        print(f"    {label:28s}  weeks={s['weeks']:4d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
              f"b&h_ret={s['bh_total_ret_pct']:+7.1f}%  ann_sharpe={s['ann_sharpe']:+.2f}  "
              f"max_dd={s['max_dd_pct']:6.1f}%  in_mkt={s['pct_weeks_in_market']:5.1f}%")

    out_path = Path(__file__).parent / "positioning_extreme_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
