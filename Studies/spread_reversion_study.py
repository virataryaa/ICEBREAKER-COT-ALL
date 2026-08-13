"""
spread_reversion_study.py — standalone (no COT data) backtest of
cross-commodity spread mean-reversion, a relative-value mechanism rather
than a directional bet.

Pairs (same "combined" groupings the dashboard already tracks):
  KC / RC   Arabica vs Robusta coffee   — both USD-denominated, clean pair
  SB / LSU  Raw vs White sugar          — both USD-denominated, clean pair
  CC / LCC  NY vs London cocoa          — CC is USD, LCC is GBP: this pair
                                          carries an unhedged FX confound
                                          (GBPUSD drift shows up in the
                                          nominal spread) — flagged, not
                                          fixed, since no FX series is
                                          loaded here.

Signal: log-spread = log(px_A) - log(px_B), z-scored against its own
ROLLING (not expanding) window — spread relationships between grades/
origins can genuinely drift over a decade, so anchoring to a rolling
window is more appropriate here than the expanding percentile used in the
positioning studies. Enter fading the spread when |z| > entry, exit when
|z| < exit (hysteresis band, avoids whipsaw at the threshold).

Run:  python spread_reversion_study.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DB_DIR     = Path(__file__).resolve().parent.parent / "Database"
ROLLEX_DIR = DB_DIR / "Rollex"
ROLLEX_MAP = {"KC": "rollex_KC.parquet", "RC": "rollex_RC.parquet",
              "CC": "rollex_CC.parquet", "LCC": "rollex_LCC.parquet",
              "SB": "rollex_SB.parquet", "LSU": "rollex_LSU.parquet"}
PAIRS = [("KC", "RC", "Arabica vs Robusta (both USD)"),
          ("SB", "LSU", "Raw vs White sugar (both USD)"),
          ("CC", "LCC", "NY vs London cocoa (FX confound: CC=USD, LCC=GBP)")]

WINDOW    = 126     # rolling z-score window, trading days (~6mo)
ENTRY_Z   = 1.5
EXIT_Z    = 0.5
N_BOOT    = 5000
COST_BPS  = 4.0      # per LEG, so a full flip costs 2x this
RNG       = np.random.default_rng(23)


def load_px(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(ROLLEX_DIR / ROLLEX_MAP[sym], columns=["rollex_px"])
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.reset_index().dropna().sort_values("Date").reset_index(drop=True)


def block_bootstrap_mean_pvalue(y: np.ndarray, block: int, n_boot=N_BOOT) -> tuple:
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
        out[label] = dict(n=len(sub), total_ret_pct=(equity[-1] - 1) * 100 if len(equity) else np.nan,
                           ann_sharpe=ann_sharpe, max_dd_pct=dd * 100 if pd.notna(dd) else np.nan)
    return out


def build_pair(sym_a: str, sym_b: str) -> pd.DataFrame:
    a, b = load_px(sym_a), load_px(sym_b)
    d = a.merge(b, on="Date", suffixes=(f"_{sym_a}", f"_{sym_b}"))
    d["spread"] = np.log(d[f"rollex_px_{sym_a}"]) - np.log(d[f"rollex_px_{sym_b}"])
    roll_mean = d["spread"].rolling(WINDOW, min_periods=WINDOW // 2).mean()
    roll_std  = d["spread"].rolling(WINDOW, min_periods=WINDOW // 2).std()
    d["z"] = (d["spread"] - roll_mean) / roll_std
    d["ret_a"] = d[f"rollex_px_{sym_a}"].pct_change()
    d["ret_b"] = d[f"rollex_px_{sym_b}"].pct_change()
    return d.dropna(subset=["z", "ret_a", "ret_b"]).reset_index(drop=True)


def spread_backtest(d: pd.DataFrame, entry=ENTRY_Z, exit=EXIT_Z, mode="fade") -> pd.DataFrame:
    """mode='fade': bet the spread reverts (short when z high, long when z low)
    mode='momentum': bet the spread keeps trending (chase the breakout)"""
    z = d["z"].values
    n = len(z)
    pos = np.zeros(n)   # +1 = long spread (long A, short B); -1 = short spread
    cur = 0.0
    sign = -1.0 if mode == "fade" else 1.0
    for i in range(n):
        if cur == 0.0:
            if z[i] > entry:
                cur = sign
            elif z[i] < -entry:
                cur = -sign
        else:
            if abs(z[i]) < exit:
                cur = 0.0
        pos[i] = cur

    pos_lag = np.r_[0.0, pos[:-1]]              # trade on yesterday's known z-score
    spread_ret = pos_lag * (d["ret_a"].values - d["ret_b"].values)
    turnover = np.r_[0, np.abs(np.diff(pos_lag))]
    cost = turnover * (2 * COST_BPS / 10000.0)   # 2 legs
    return pd.DataFrame(dict(Date=d["Date"].values, ret=spread_ret - cost, pos=pos_lag, z=z))


def window_consistency(bt: pd.DataFrame, n_windows=4) -> str:
    """Split the backtest into n_windows equal chunks and report Sharpe in
    each — checks whether the edge is spread across time (real, repeatable)
    or concentrated in one or two episodes (backtest artifact)."""
    bt = bt.set_index("Date")
    n = len(bt)
    chunk = n // n_windows
    parts = []
    for i in range(n_windows):
        sub = bt.iloc[i * chunk: (i + 1) * chunk if i < n_windows - 1 else n]
        r = sub["ret"].values
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        parts.append(f"{sub.index[0].date()}..{sub.index[-1].date()}: Sharpe={sharpe:+.2f}")
    return "  |  ".join(parts)


def main():
    all_rows = []
    for sym_a, sym_b, note in PAIRS:
        d = build_pair(sym_a, sym_b)
        print(f"\n{'='*70}\n{sym_a}/{sym_b}  -  {note}\n{len(d)} trading days, "
              f"{d['Date'].min().date()} to {d['Date'].max().date()}\n{'='*70}")

        for mode in ("fade", "momentum"):
            print(f"\n  --- mode = {mode} ---")
            print(f"    {'entry':>6s} {'exit':>6s} {'full_Sharpe':>12s} {'IS_Sharpe':>10s} "
                  f"{'OOS_Sharpe':>11s} {'ann_ret%':>9s} {'p':>7s}")
            for entry, exit in [(1.0, 0.25), (1.5, 0.5), (2.0, 0.5), (2.5, 1.0)]:
                bt2 = spread_backtest(d, entry=entry, exit=exit, mode=mode)
                s2 = backtest_stats(bt2)
                obs2, p2 = block_bootstrap_mean_pvalue(bt2["ret"].values, block=WINDOW, n_boot=2000)
                print(f"    {entry:6.1f} {exit:6.2f} {s2['full']['ann_sharpe']:12.2f} "
                      f"{s2['in-sample (first 70%)']['ann_sharpe']:10.2f} "
                      f"{s2['out-of-sample (last 30%)']['ann_sharpe']:11.2f} {obs2*252*100:9.2f} {p2:7.4f}")
                all_rows.append(dict(pair=f"{sym_a}/{sym_b}", mode=mode, note=f"entry={entry},exit={exit}",
                                      ann_ret_pct=obs2 * 252 * 100, p=p2,
                                      full_sharpe=s2["full"]["ann_sharpe"],
                                      is_sharpe=s2["in-sample (first 70%)"]["ann_sharpe"],
                                      oos_sharpe=s2["out-of-sample (last 30%)"]["ann_sharpe"]))
                if mode == "momentum" and entry == 1.5:
                    print(f"      Window consistency: {window_consistency(bt2)}")

    df = pd.DataFrame(all_rows)
    df["p_bh"] = bh_correct(df["p"].tolist())
    print(f"\n\n{'='*70}\nALL RESULTS  (p_bh = Benjamini-Hochberg corrected across all "
          f"pair/threshold combos tested)\n{'='*70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df[["pair", "mode", "note", "ann_ret_pct", "full_sharpe", "is_sharpe", "oos_sharpe", "p", "p_bh"]]
              .round(3).to_string(index=False))

    sig = df[df["p_bh"] <= 0.10]
    print(f"\n{'='*70}\nSURVIVES q<=0.10 after FDR correction: {len(sig)} / {len(df)}\n{'='*70}")

    out_path = Path(__file__).parent / "spread_reversion_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
