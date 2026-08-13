"""
carry_study.py — standalone (no Streamlit, no COT data) backtest of the
commodity carry / roll-yield factor across KC, CC, SB, CT, RC, LCC.

  Study C1  Time-series carry — per commodity, long when in backwardation
                                 (Roll_Yield_1yr > 0), short/flat when in
                                 contango. Individually + pooled portfolio.
  Study C2  Cross-sectional carry — each rebalance, long the highest-carry
                                     markets, short the lowest-carry ones,
                                     dollar-neutral long/short portfolio.

Roll_Yield_1yr = Spot/OneYr - 1, computed from that day's own prices only
(no look-ahead by construction). Run:  python carry_study.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DB_DIR    = Path(__file__).resolve().parent.parent / "Database"
CARRY_FILE = DB_DIR / "RollYield" / "roll_yield_data.parquet"
SYMBOLS    = ["KC", "CC", "SB", "CT", "RC", "LCC"]

N_BOOT    = 5000
COST_BPS  = 4.0
RNG       = np.random.default_rng(11)


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════
def load_carry(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(CARRY_FILE)
    df = df[df["Commodity"] == sym].sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Spot", "Roll_Yield_1yr"]).reset_index(drop=True)
    return df[["Date", "Spot", "Roll_Yield_1yr"]]


def forward_return(px: pd.Series, n: int) -> np.ndarray:
    return (px.shift(-n).values / px.values - 1.0) * 100.0


# ══════════════════════════════════════════════════════════════════════════
# Significance tests
# ══════════════════════════════════════════════════════════════════════════
def block_bootstrap_pvalue(mask: np.ndarray, y: np.ndarray, block: int, n_boot=N_BOOT) -> tuple:
    valid = ~np.isnan(y) & ~np.isnan(mask.astype(float))
    y, mask = y[valid], mask[valid]
    if mask.sum() < 20 or (~mask).sum() < 20:
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


def block_bootstrap_mean_pvalue(y: np.ndarray, block: int, n_boot=N_BOOT) -> tuple:
    """Test H0: population mean of y is 0, via a null-centred moving-block
    bootstrap (needed for the cross-sectional portfolio, which is a single
    return series, not a two-group comparison)."""
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 30:
        return np.nan, np.nan, np.nan
    obs = float(y.mean())
    y0 = y - obs
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.integers(0, n - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        means[b] = y0[idx].mean()
    p = float((np.abs(means) >= abs(obs)).mean())
    lo, hi = np.percentile(means + obs, [2.5, 97.5])
    return obs, p, (lo, hi)


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
        day_sharpe = r.mean() / std if std > 0 else np.nan
        ann_sharpe = day_sharpe * np.sqrt(252)
        equity = (1 + r).cumprod()
        dd = (equity / np.maximum.accumulate(equity) - 1).min() if len(equity) else np.nan
        bh_equity = (1 + sub["bh_ret"].values).cumprod() if "bh_ret" in sub else None
        out[label] = dict(
            n=len(sub), total_ret_pct=(equity[-1] - 1) * 100 if len(equity) else np.nan,
            bh_total_ret_pct=(bh_equity[-1] - 1) * 100 if bh_equity is not None and len(bh_equity) else np.nan,
            ann_sharpe=ann_sharpe, max_dd_pct=dd * 100 if pd.notna(dd) else np.nan)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Study C1 — Time-series carry per commodity
# ══════════════════════════════════════════════════════════════════════════
def study_c1_timeseries(d: pd.DataFrame, horizons=(5, 10, 20, 60)) -> tuple:
    carry = d["Roll_Yield_1yr"].values
    rows = []
    for h in horizons:
        fwd = forward_return(d["Spot"], h)
        backw = carry > 0
        diff, p, ci = block_bootstrap_pvalue(backw, fwd, block=max(h * 2, 10))
        rows.append(dict(study="C1", bucket="backwardation (carry>0)", horizon=h,
                          n=int(np.nansum(backw)), mean_diff_pct=diff, p=p, ci=ci))

    # Backtest: long when in backwardation, short when in contango, signal
    # lagged 1 trading day (act on yesterday's known carry).
    sig = np.where(carry > 0, 1.0, -1.0)
    px = d["Spot"].values
    ret_1d = np.r_[np.nan, px[1:] / px[:-1] - 1.0]
    ret_1d = np.nan_to_num(ret_1d)
    pos = np.r_[np.nan, sig[:-1]]
    pos = np.where(np.isnan(pos), 0, pos)

    strat_ret = pos * ret_1d
    cost = np.r_[0, np.abs(np.diff(pos))] * (COST_BPS / 10000.0)
    bt = pd.DataFrame(dict(Date=d["Date"].values, ret=strat_ret - cost, bh_ret=ret_1d, pos=pos))
    return rows, bt


def pooled_timeseries_backtest(data_by_sym: dict) -> pd.DataFrame:
    per_sym = {}
    for sym, d in data_by_sym.items():
        carry = d["Roll_Yield_1yr"].values
        sig = np.where(carry > 0, 1.0, -1.0)
        px = d["Spot"].values
        ret_1d = np.nan_to_num(np.r_[np.nan, px[1:] / px[:-1] - 1.0])
        pos = np.r_[np.nan, sig[:-1]]
        pos = np.where(np.isnan(pos), 0, pos)
        strat_ret = pos * ret_1d
        cost = np.r_[0, np.abs(np.diff(pos))] * (COST_BPS / 10000.0)
        per_sym[sym] = pd.DataFrame(dict(Date=d["Date"].values, ret=strat_ret - cost,
                                          bh_ret=ret_1d)).rename(
            columns={"ret": f"ret_{sym}", "bh_ret": f"bh_{sym}"})
    merged = None
    for bt in per_sym.values():
        merged = bt if merged is None else merged.merge(bt, on="Date", how="inner")
    ret_cols = [c for c in merged.columns if c.startswith("ret_")]
    bh_cols  = [c for c in merged.columns if c.startswith("bh_")]
    return pd.DataFrame(dict(Date=merged["Date"], ret=merged[ret_cols].mean(axis=1),
                              bh_ret=merged[bh_cols].mean(axis=1)))


# ══════════════════════════════════════════════════════════════════════════
# Study C2 — Cross-sectional carry factor
# Every REBAL trading days: long the top LEGN carry markets, short the
# bottom LEGN, equal weight within each leg, dollar neutral. Classic
# academic construction of the carry factor (vs C1's per-market timing).
# ══════════════════════════════════════════════════════════════════════════
def study_c2_cross_sectional(data_by_sym: dict, rebal=5, leg_n=2, horizons=(5, 10, 20, 60)) -> tuple:
    merged = None
    for sym, d in data_by_sym.items():
        sub = d[["Date", "Spot", "Roll_Yield_1yr"]].rename(
            columns={"Spot": f"px_{sym}", "Roll_Yield_1yr": f"cy_{sym}"})
        merged = sub if merged is None else merged.merge(sub, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)

    px_cols = [f"px_{s}" for s in data_by_sym]
    cy_cols = [f"cy_{s}" for s in data_by_sym]
    syms = list(data_by_sym.keys())

    daily_ret = merged[px_cols].pct_change().fillna(0.0)
    daily_ret.columns = syms

    n = len(merged)
    weights = pd.DataFrame(0.0, index=merged.index, columns=syms)
    rebal_idx = list(range(0, n, rebal))
    for i in rebal_idx:
        cy_today = merged.loc[i, cy_cols].values.astype(float)
        order = np.argsort(cy_today)             # ascending: lowest carry first
        short_syms = [syms[j] for j in order[:leg_n]]
        long_syms  = [syms[j] for j in order[-leg_n:]]
        w = pd.Series(0.0, index=syms)
        w[long_syms]  = 1.0 / leg_n
        w[short_syms] = -1.0 / leg_n
        end = min(i + rebal, n)
        weights.iloc[i:end] = w.values

    weights_lag = weights.shift(1).fillna(0.0)     # trade on yesterday's known ranking
    port_ret = (weights_lag.values * daily_ret.values).sum(axis=1)
    turnover = np.abs(weights_lag.diff().fillna(0.0)).sum(axis=1).values
    cost = turnover * (COST_BPS / 10000.0)
    port_ret_net = port_ret - cost

    bt = pd.DataFrame(dict(Date=merged["Date"].values, ret=port_ret_net))

    rows = []
    for h in horizons:
        fwd_win = pd.Series(port_ret_net).rolling(h).sum().shift(-h + 1).values * 100.0
        obs, p, ci = block_bootstrap_mean_pvalue(fwd_win, block=max(h * 2, 10))
        rows.append(dict(study="C2", bucket=f"long-short carry portfolio (top{leg_n} vs bottom{leg_n})",
                          horizon=h, n=int(np.sum(~np.isnan(fwd_win))), mean_diff_pct=obs, p=p, ci=ci))
    return rows, bt


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════
def main():
    all_rows = []
    data_by_sym = {}
    for sym in SYMBOLS:
        d = load_carry(sym)
        data_by_sym[sym] = d
        print(f"\n{'='*70}\n{sym} - {len(d)} trading days, {d['Date'].min().date()} to {d['Date'].max().date()}\n{'='*70}")

        r1, bt1 = study_c1_timeseries(d)
        for r in r1:
            r["symbol"] = sym
        all_rows += r1

        stats1 = backtest_stats(bt1)
        print(f"\n  Study C1 backtest ({sym}, long/short on sign(carry), {COST_BPS}bps cost, signal lagged 1d):")
        for label, s in stats1.items():
            print(f"    {label:28s}  n={s['n']:5d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
                  f"b&h_ret={s['bh_total_ret_pct']:+7.1f}%  ann_sharpe={s['ann_sharpe']:+.2f}  "
                  f"max_dd={s['max_dd_pct']:6.1f}%")

    bt1_pooled = pooled_timeseries_backtest(data_by_sym)
    stats1_pooled = backtest_stats(bt1_pooled)
    print(f"\n\n{'='*70}\nPOOLED Study C1 (equal-weight KC/CC/SB/CT/RC/LCC, {COST_BPS}bps cost)\n{'='*70}")
    for label, s in stats1_pooled.items():
        print(f"    {label:28s}  n={s['n']:5d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
              f"b&h_ret={s['bh_total_ret_pct']:+7.1f}%  ann_sharpe={s['ann_sharpe']:+.2f}  "
              f"max_dd={s['max_dd_pct']:6.1f}%")

    r2, bt2 = study_c2_cross_sectional(data_by_sym)
    for r in r2:
        r["symbol"] = "CROSS-SECTIONAL(6 markets)"
    all_rows += r2

    stats2 = backtest_stats(bt2)
    print(f"\n\n{'='*70}\nStudy C2 backtest (cross-sectional carry, top2 vs bottom2 of 6, "
          f"5d rebal, {COST_BPS}bps cost, signal lagged 1d)\n{'='*70}")
    for label, s in stats2.items():
        print(f"    {label:28s}  n={s['n']:5d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
              f"ann_sharpe={s['ann_sharpe']:+.2f}  max_dd={s['max_dd_pct']:6.1f}%")

    df = pd.DataFrame(all_rows)
    df["p_bh"] = bh_correct(df["p"].tolist())

    print(f"\n\n{'='*70}\nBUCKET / PORTFOLIO SIGNIFICANCE TESTS  (p from {N_BOOT} moving-block "
          f"bootstraps; p_bh = Benjamini-Hochberg corrected across all tests)\n{'='*70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df[["symbol", "study", "bucket", "horizon", "n", "mean_diff_pct", "p", "p_bh"]]
              .round(3).to_string(index=False))

    sig = df[df["p_bh"] <= 0.10]
    print(f"\n\n{'='*70}\nSURVIVES q<=0.10 after FDR correction: {len(sig)} / {len(df)} tests\n{'='*70}")
    if len(sig):
        print(sig[["symbol", "study", "bucket", "horizon", "mean_diff_pct", "p_bh"]].to_string(index=False))
    else:
        print("None. Neither the time-series nor cross-sectional carry factor clears a "
              "false-discovery-corrected bar on this data.")

    out_path = Path(__file__).parent / "carry_study_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
