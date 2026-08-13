"""
trend_following_study.py — standalone (no COT data at all) backtest of
classic time-series momentum / trend-following across all seven
commodities: KC, CC, SB, CT, RC, LCC, LSU.

Signal: sign of the trailing L-day return (Moskowitz/Ooi/Pedersen-style
time-series momentum). Position = that sign, lagged 1 day, held until it
flips. Tested individually per commodity and pooled as an equal-weight
diversified trend portfolio (the standard construction — trend's main
edge historically comes from pooling many weakly-correlated trends, not
any single market).

Data: Database/Rollex/rollex_{SYM}.parquet, rollex_px (continuous,
roll-adjusted daily close) — same series already used elsewhere in this
project, no COT dependency.

Run:  python trend_following_study.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DB_DIR     = Path(__file__).resolve().parent.parent / "Database"
ROLLEX_DIR = DB_DIR / "Rollex"
ROLLEX_MAP = {"KC": "rollex_KC.parquet", "CC": "rollex_CC.parquet",
              "SB": "rollex_SB.parquet", "CT": "rollex_CT.parquet",
              "RC": "rollex_RC.parquet", "LCC": "rollex_LCC.parquet",
              "LSU": "rollex_LSU.parquet"}

LOOKBACKS = (21, 63, 126, 252)     # ~1mo, 3mo, 6mo, 12mo trading days
EVAL_HORIZONS = (5, 21, 63)        # forward window for the bucket significance test
N_BOOT   = 5000
COST_BPS = 4.0
RNG      = np.random.default_rng(17)


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════
def load_commodity(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(ROLLEX_DIR / ROLLEX_MAP[sym], columns=["rollex_px"])
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.reset_index().dropna(subset=["rollex_px"]).sort_values("Date").reset_index(drop=True)
    return df


def forward_return(px: pd.Series, n: int) -> np.ndarray:
    return (px.shift(-n).values / px.values - 1.0) * 100.0


# ══════════════════════════════════════════════════════════════════════════
# Significance tests (same machinery as the other Studies scripts)
# ══════════════════════════════════════════════════════════════════════════
def block_bootstrap_pvalue(mask: np.ndarray, y: np.ndarray, block: int, n_boot=N_BOOT) -> tuple:
    valid = ~np.isnan(y) & ~np.isnan(mask.astype(float))
    y, mask = y[valid], mask[valid]
    if mask.sum() < 20 or (~mask).sum() < 20:
        return np.nan, np.nan
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
    return obs_diff, p


def pooled_block_bootstrap_pvalue(groups: list, block: int, n_boot=N_BOOT) -> tuple:
    clean = []
    for mask, y in groups:
        valid = ~np.isnan(y) & ~np.isnan(mask.astype(float))
        m, yv = mask[valid], y[valid]
        if len(yv) > 0:
            clean.append((m, yv))
    if not clean:
        return np.nan, np.nan, 0
    all_mask = np.concatenate([m for m, _ in clean])
    all_y = np.concatenate([y for _, y in clean])
    if all_mask.sum() < 20 or (~all_mask).sum() < 20:
        return np.nan, np.nan, int(all_mask.sum())
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
    return obs_diff, p, int(all_mask.sum())


def block_bootstrap_mean_pvalue(y: np.ndarray, block: int, n_boot=N_BOOT) -> tuple:
    """Test H0: mean portfolio return is 0, via null-centred moving-block
    bootstrap. More direct than the bucket test for judging the actual
    backtested portfolio (bucket test compares two group means of forward
    returns; this tests the realised P&L series itself)."""
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 30:
        return np.nan, np.nan
    obs = float(y.mean())
    y0 = y - obs
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        means[b] = y0[idx].mean()
    p = float((np.abs(means) >= abs(obs)).mean())
    return obs, p


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
        ann_sharpe = (r.mean() / std * np.sqrt(252)) if std > 0 else np.nan
        equity = (1 + r).cumprod()
        dd = (equity / np.maximum.accumulate(equity) - 1).min() if len(equity) else np.nan
        bh_equity = (1 + sub["bh_ret"].values).cumprod() if "bh_ret" in sub else None
        out[label] = dict(n=len(sub), total_ret_pct=(equity[-1] - 1) * 100 if len(equity) else np.nan,
                           bh_total_ret_pct=(bh_equity[-1] - 1) * 100 if bh_equity is not None and len(bh_equity) else np.nan,
                           ann_sharpe=ann_sharpe, max_dd_pct=dd * 100 if pd.notna(dd) else np.nan)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Time-series momentum
# ══════════════════════════════════════════════════════════════════════════
def trend_signal(d: pd.DataFrame, lookback: int) -> np.ndarray:
    px = d["rollex_px"].values
    sig = np.full(len(px), np.nan)
    sig[lookback:] = np.sign(px[lookback:] / px[:-lookback] - 1.0)
    return sig


def bucket_tests(data_by_sym: dict) -> list:
    rows = []
    for lb in LOOKBACKS:
        for h in EVAL_HORIZONS:
            groups = []
            for sym, d in data_by_sym.items():
                sig = trend_signal(d, lb)
                fwd = forward_return(d["rollex_px"], h)
                up = sig > 0
                diff, p = block_bootstrap_pvalue(up, fwd, block=max(h * 2, 10))
                rows.append(dict(symbol=sym, lookback=lb, horizon=h,
                                  n=int(np.nansum(up)), mean_diff_pct=diff, p=p))
                groups.append((up, fwd))
            diff, p, n = pooled_block_bootstrap_pvalue(groups, block=max(h * 2, 10))
            rows.append(dict(symbol="POOLED(7)", lookback=lb, horizon=h, n=n, mean_diff_pct=diff, p=p))
    return rows


def trend_backtest(d: pd.DataFrame, lookback: int) -> pd.DataFrame:
    sig = trend_signal(d, lookback)
    px = d["rollex_px"].values
    ret_1d = np.nan_to_num(np.r_[np.nan, px[1:] / px[:-1] - 1.0])
    pos = np.r_[np.nan, sig[:-1]]
    pos = np.where(np.isnan(pos), 0, pos)
    strat_ret = pos * ret_1d
    cost = np.r_[0, np.abs(np.diff(pos))] * (COST_BPS / 10000.0)
    return pd.DataFrame(dict(Date=d["Date"].values, ret=strat_ret - cost, bh_ret=ret_1d, pos=pos))


def pooled_trend_backtest(data_by_sym: dict, lookback: int) -> pd.DataFrame:
    per_sym = {}
    for sym, d in data_by_sym.items():
        bt = trend_backtest(d, lookback).drop(columns=["pos"])
        per_sym[sym] = bt.rename(columns={"ret": f"ret_{sym}", "bh_ret": f"bh_{sym}"})
    merged = None
    for bt in per_sym.values():
        merged = bt if merged is None else merged.merge(bt, on="Date", how="inner")
    ret_cols = [c for c in merged.columns if c.startswith("ret_")]
    bh_cols  = [c for c in merged.columns if c.startswith("bh_")]
    return pd.DataFrame(dict(Date=merged["Date"], ret=merged[ret_cols].mean(axis=1),
                              bh_ret=merged[bh_cols].mean(axis=1)))


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════
def main():
    data_by_sym = {}
    for sym in ["KC", "CC", "SB", "CT", "RC", "LCC", "LSU"]:
        d = load_commodity(sym)
        data_by_sym[sym] = d
        print(f"{sym:5s}  {len(d):5d} trading days  {d['Date'].min().date()} to {d['Date'].max().date()}")

    rows = bucket_tests(data_by_sym)
    df = pd.DataFrame(rows)
    df["p_bh"] = bh_correct(df["p"].tolist())

    print(f"\n{'='*70}\nBUCKET TESTS: trend-up vs forward return, per symbol and pooled\n{'='*70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df[["symbol", "lookback", "horizon", "n", "mean_diff_pct", "p", "p_bh"]]
              .round(3).to_string(index=False))

    sig = df[df["p_bh"] <= 0.10]
    print(f"\n{'='*70}\nSURVIVES q<=0.10 after FDR correction: {len(sig)} / {len(df)}\n{'='*70}")
    if len(sig):
        print(sig[["symbol", "lookback", "horizon", "mean_diff_pct", "p_bh"]].to_string(index=False))

    print(f"\n\n{'='*70}\nTREND-FOLLOWING BACKTEST  ({COST_BPS}bps cost, signal lagged 1 trading day)\n{'='*70}")
    for lb in LOOKBACKS:
        print(f"\n  --- lookback = {lb}d ---")
        for sym, d in data_by_sym.items():
            stats = backtest_stats(trend_backtest(d, lb))
            f = stats["full"]
            oos = stats["out-of-sample (last 30%)"]
            print(f"    {sym:5s}  full: strat={f['total_ret_pct']:+7.1f}%  b&h={f['bh_total_ret_pct']:+7.1f}%  "
                  f"sharpe={f['ann_sharpe']:+.2f}  dd={f['max_dd_pct']:6.1f}%   |   "
                  f"OOS: strat={oos['total_ret_pct']:+7.1f}%  sharpe={oos['ann_sharpe']:+.2f}")

        pooled_bt = pooled_trend_backtest(data_by_sym, lb)
        pooled_stats = backtest_stats(pooled_bt)
        print(f"    {'POOLED(7)':5s}:")
        for label, s in pooled_stats.items():
            print(f"      {label:28s}  n={s['n']:5d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
                  f"b&h_ret={s['bh_total_ret_pct']:+7.1f}%  ann_sharpe={s['ann_sharpe']:+.2f}  "
                  f"max_dd={s['max_dd_pct']:6.1f}%")

        obs, p = block_bootstrap_mean_pvalue(pooled_bt["ret"].values, block=max(lb, 20))
        print(f"      Direct test: is the pooled portfolio's mean daily return != 0?  "
              f"annualized={obs*252*100:+.2f}%  p={p:.4f}")

    out_path = Path(__file__).parent / "trend_following_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
