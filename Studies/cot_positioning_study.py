"""
cot_positioning_study.py — standalone (no Streamlit) backtest of three COT
positioning hypotheses, run independently of the dashboard.

  Study 1  Extremity mean-reversion   — crowded Spec Long/Short vs its own
                                         history predicts forward returns.
  Study 2  Smart-money divergence     — Commercial vs Spec net positioning
                                         gap predicts forward returns; also
                                         run as an actual long/flat/short
                                         backtest with costs.
  Study 4  Crowding + low-vol squeeze — extreme positioning during a vol
                                         lull precedes larger forward moves.
  Study 5  Spec Saturation Index      — gross Spec Long at a point-in-time
                                         historical high, still adding this
                                         week, but price doesn't confirm ->
                                         buying exhaustion -> forced long
                                         liquidation and a decline over the
                                         following weeks. Also run as an
                                         actual short-only backtest.

Run:  python cot_positioning_study.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sps

# ── Paths (mirrors Dashboard/cot_app.py) ────────────────────────────────────
DB_DIR     = Path(__file__).resolve().parent.parent / "Database"
CIT_FILE   = DB_DIR / "cot_cit.parquet"
ROLLEX_DIR = DB_DIR / "Rollex"
ROLLEX_MAP = {"KC": "rollex_KC.parquet", "CC": "rollex_CC.parquet",
              "SB": "rollex_SB.parquet", "CT": "rollex_CT.parquet"}

MIN_HISTORY_WEEKS = 104     # need 2 years of history before a percentile counts
N_BOOT            = 5000    # bootstrap resamples for significance
BLOCK             = 8       # bootstrap block length (weeks) — approximates the
                             # autocorrelation induced by overlapping fwd returns
COST_BPS          = 4.0     # one-way cost assumption per position change (futures)
RNG               = np.random.default_rng(7)


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════
def load_commodity(sym: str) -> pd.DataFrame:
    cit = pd.read_parquet(CIT_FILE)
    cit = cit[cit["Commodity"] == sym].sort_values("Date").reset_index(drop=True)
    cit["Date"] = pd.to_datetime(cit["Date"])

    rx = pd.read_parquet(ROLLEX_DIR / ROLLEX_MAP[sym], columns=["rollex_px"])
    rx.index = pd.to_datetime(rx.index)
    rx.index.name = "Date"
    rx = rx.reset_index().sort_values("Date")

    d = pd.merge_asof(cit, rx, on="Date", direction="backward")
    d = d.dropna(subset=["rollex_px", "Spec Long", "Spec Short",
                          "Comm Long", "Comm Short"]).reset_index(drop=True)
    return d


# ══════════════════════════════════════════════════════════════════════════
# Point-in-time percentile — expanding window, NO look-ahead.
# Row i's percentile only ever uses rows [0..i]. This is what the earlier
# dashboard chart got wrong (it ranked against the FULL sample, which bakes
# in future information a live trader would never have had that week).
# ══════════════════════════════════════════════════════════════════════════
def expanding_percentile(s: pd.Series, min_periods=MIN_HISTORY_WEEKS) -> np.ndarray:
    vals = s.values.astype(float)
    out = np.full(len(vals), np.nan)
    for i in range(min_periods - 1, len(vals)):
        window = vals[: i + 1]
        out[i] = (window <= vals[i]).mean() * 100.0
    return out


def forward_return(px: pd.Series, n: int) -> np.ndarray:
    return (px.shift(-n).values / px.values - 1.0) * 100.0


# ══════════════════════════════════════════════════════════════════════════
# Significance: moving-block bootstrap of the mean-difference between a
# bucket and the unconditional (all-week) mean. Overlapping forward returns
# make a plain t-test overstate significance, so we resample in blocks
# rather than i.i.d. rows.
# ══════════════════════════════════════════════════════════════════════════
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


def bh_correct(pvals: list) -> list:
    """Benjamini-Hochberg FDR correction across all tests run in this script."""
    pvals = np.array(pvals, dtype=float)
    n = np.sum(~np.isnan(pvals))
    order = np.argsort(np.where(np.isnan(pvals), np.inf, pvals))
    ranks = np.empty(len(pvals)); ranks[order] = np.arange(1, len(pvals) + 1)
    adj = pvals * n / np.where(ranks == 0, 1, ranks)
    return list(np.minimum.accumulate(adj[order][::-1])[::-1][np.argsort(order)])


# ══════════════════════════════════════════════════════════════════════════
# Study 1 — Extremity mean-reversion
# ══════════════════════════════════════════════════════════════════════════
def study1_extremity(d: pd.DataFrame, horizons=(1, 4, 8)) -> list:
    rows = []
    pctL = expanding_percentile(d["Spec Long"])
    pctS = expanding_percentile(d["Spec Short"])
    for leg, pct, tag in [("Spec Long", pctL, "crowded LONG"), ("Spec Short", pctS, "crowded SHORT")]:
        for h in horizons:
            fwd = forward_return(d["rollex_px"], h)
            high = pct >= 90
            diff, p, ci = block_bootstrap_pvalue(high, fwd)
            rows.append(dict(study="S1", leg=leg, bucket=f"{tag} (pctl>=90)", horizon=h,
                              n=int(np.nansum(high)), mean_diff_pct=diff, p=p, ci=ci))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Study 2 — Smart-money (Commercial) vs Spec divergence
# ══════════════════════════════════════════════════════════════════════════
def study2_divergence(d: pd.DataFrame, horizons=(1, 4, 8)) -> tuple:
    spec_net = d["Spec Long"] - d["Spec Short"]
    comm_net = d["Comm Long"] - d["Comm Short"]
    pct_spec = expanding_percentile(spec_net)
    pct_comm = expanding_percentile(comm_net)
    divergence = pct_spec - pct_comm     # + = spec crowded long while commercials aren't

    rows = []
    for h in horizons:
        fwd = forward_return(d["rollex_px"], h)
        bull = divergence <= -60   # commercials relatively more long than spec -> bullish (contrarian)
        bear = divergence >= 60    # spec crowded long relative to commercials -> bearish
        for tag, m in [("bullish (comm > spec, div<=-60)", bull), ("bearish (spec > comm, div>=60)", bear)]:
            diff, p, ci = block_bootstrap_pvalue(m, fwd)
            rows.append(dict(study="S2", leg="divergence", bucket=tag, horizon=h,
                              n=int(np.nansum(m)), mean_diff_pct=diff, p=p, ci=ci))

    # ── Actual backtest: long when bullish signal, short when bearish, flat else.
    # Signal and position are LAGGED by one week — position taken based on this
    # week's divergence trades the NEXT week's return, so nothing is peeked at.
    sig = np.where(divergence <= -60, 1, np.where(divergence >= 60, -1, 0))
    px = d["rollex_px"].values
    ret_1w = np.r_[np.nan, px[1:] / px[:-1] - 1.0]
    pos = np.r_[np.nan, sig[:-1]]                       # position held INTO week t, decided at t-1
    pos = np.where(np.isnan(pos), 0, pos)

    strat_ret = pos * np.nan_to_num(ret_1w)
    turned = np.r_[0, np.abs(np.diff(pos))]             # position changes -> cost
    cost = turned * (COST_BPS / 10000.0)
    strat_ret_net = strat_ret - cost

    bt = pd.DataFrame(dict(Date=d["Date"].values, ret=strat_ret_net,
                            bh_ret=np.nan_to_num(ret_1w), pos=pos))
    return rows, bt


def backtest_stats(bt: pd.DataFrame, split_frac=0.7) -> dict:
    n = len(bt)
    cut = int(n * split_frac)
    out = {}
    for label, sub in [("full", bt), ("in-sample (first 70%)", bt.iloc[:cut]),
                        ("out-of-sample (last 30%)", bt.iloc[cut:])]:
        r = sub["ret"].values
        wk_sharpe = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
        ann_sharpe = wk_sharpe * np.sqrt(52)
        equity = (1 + r).cumprod()
        dd = (equity / np.maximum.accumulate(equity) - 1).min()
        bh_equity = (1 + sub["bh_ret"].values).cumprod()
        out[label] = dict(
            weeks=len(sub), total_ret_pct=(equity[-1] - 1) * 100 if len(equity) else np.nan,
            bh_total_ret_pct=(bh_equity[-1] - 1) * 100 if len(bh_equity) else np.nan,
            ann_sharpe=ann_sharpe, max_dd_pct=dd * 100,
            pct_weeks_in_market=(sub["pos"] != 0).mean() * 100)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Study 4 — Crowding + low-vol squeeze setup
# ══════════════════════════════════════════════════════════════════════════
def study4_squeeze(d: pd.DataFrame, horizons=(1, 4, 8)) -> list:
    pctL = expanding_percentile(d["Spec Long"])
    pctS = expanding_percentile(d["Spec Short"])
    crowding = np.fmax(pctL, pctS)   # elementwise max, NaN only where BOTH are NaN (no warning)

    ret_1w = d["rollex_px"].pct_change() * 100
    vol8 = ret_1w.rolling(8, min_periods=8).std()
    vol_pct = expanding_percentile(vol8)

    squeeze = (crowding >= 85) & (vol_pct <= 25)

    rows = []
    for h in horizons:
        fwd_abs_ret = np.abs(forward_return(d["rollex_px"], h))
        diff, p, ci = block_bootstrap_pvalue(squeeze, fwd_abs_ret)
        rows.append(dict(study="S4", leg="squeeze", bucket=f"crowd>=85 & vol_pctl<=25 (n_setup={int(np.nansum(squeeze))})",
                          horizon=h, n=int(np.nansum(squeeze)), mean_diff_pct=diff, p=p, ci=ci,
                          note="target = |fwd return|, not signed"))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Study 5 — Spec Saturation Index
# Gross Spec Long at a point-in-time historical high, still getting fresh
# buying THIS week (Delta Long > 0), but price fails to confirm (DeltaPx<=0).
# Hypothesis: buying exhaustion / distribution -> forced long liquidation
# and a disorderly decline over the following weeks.
# ══════════════════════════════════════════════════════════════════════════
def study5_saturation(d: pd.DataFrame, horizons=(1, 2, 4, 8), hold=4) -> tuple:
    pctL   = expanding_percentile(d["Spec Long"])
    dLong  = d["Spec Long"].diff().values
    dPx    = (d["rollex_px"].pct_change() * 100).values

    saturated = (pctL >= 90) & (dLong > 0) & (dPx <= 0)

    rows = []
    for h in horizons:
        fwd = forward_return(d["rollex_px"], h)
        diff, p, ci = block_bootstrap_pvalue(saturated, fwd)
        rows.append(dict(study="S5", leg="saturation",
                          bucket=f"pctl>=90 & dLong>0 & dPx<=0 (n={int(np.nansum(saturated))})",
                          horizon=h, n=int(np.nansum(saturated)), mean_diff_pct=diff, p=p, ci=ci,
                          note="target = fwd price return"))

        # Mechanism check: does gross Spec Long actually FALL in the following
        # weeks after a saturation signal, confirming the liquidation story
        # (rather than just price falling for some unrelated reason)?
        fwd_dlong = (d["Spec Long"].shift(-h).values - d["Spec Long"].values) / 1000.0
        diff_l, p_l, ci_l = block_bootstrap_pvalue(saturated, fwd_dlong)
        rows.append(dict(study="S5", leg="saturation_mechanism",
                          bucket=f"pctl>=90 & dLong>0 & dPx<=0 (n={int(np.nansum(saturated))})",
                          horizon=h, n=int(np.nansum(saturated)), mean_diff_pct=diff_l, p=p_l, ci=ci_l,
                          note="target = fwd chg in gross Spec Long (k lots), NOT price"))

    # ── Actual backtest: short for `hold` weeks after a saturation signal,
    # flat otherwise. Entry is lagged 1 week (signal known Friday when COT
    # prints, earliest trade is the following week), no pyramiding of an
    # already-open short if a fresh signal fires mid-hold.
    n = len(d)
    pos = np.zeros(n)
    trigger_idx = np.where(saturated)[0]
    for i in trigger_idx:
        entry = i + 1
        if entry >= n:
            continue
        pos[entry: min(entry + hold, n)] = -1.0

    px = d["rollex_px"].values
    ret_1w = np.r_[np.nan, px[1:] / px[:-1] - 1.0]
    ret_1w = np.nan_to_num(ret_1w)

    strat_ret = pos * ret_1w
    turned = np.r_[0, np.abs(np.diff(pos))]
    cost = turned * (COST_BPS / 10000.0)
    strat_ret_net = strat_ret - cost

    bt = pd.DataFrame(dict(Date=d["Date"].values, ret=strat_ret_net, bh_ret=ret_1w, pos=pos))
    return rows, bt


# ══════════════════════════════════════════════════════════════════════════
# Pooled significance: n=17-38 events per commodity in Study 5 is too thin
# to trust. This resamples blocks WITHIN each commodity (preserving its own
# autocorrelation) then pools the resample across all four for one combined
# test — same idea as a fixed-effects meta-analysis, more power than four
# separate underpowered tests.
# ══════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════
# Study 5b — Spec Saturation Index, loosened filters + pooled across
# KC/CC/SB/CT. Study 5's exact filter (pctl>=90, dPx<=0) only fires 17-38x
# per commodity — too sparse to trust either way. This sweeps looser
# extremity cuts (>=80/85/90) and a relative price filter (bottom-third of
# that week's own price-move history, instead of a hard zero cutoff), and
# pools events across all four commodities for real statistical power.
# ══════════════════════════════════════════════════════════════════════════
def saturation_mask(d: pd.DataFrame, extremity_thresh: float, price_mode: str) -> np.ndarray:
    pctL  = expanding_percentile(d["Spec Long"])
    dLong = d["Spec Long"].diff().values
    dPx_s = d["rollex_px"].pct_change() * 100
    if price_mode == "le0":
        px_ok = (dPx_s <= 0).values
    elif price_mode == "bottom_third":
        px_ok = (expanding_percentile(dPx_s) <= 33)
    else:
        raise ValueError(price_mode)
    return (pctL >= extremity_thresh) & (dLong > 0) & px_ok


def study5b_pooled_sweep(data_by_sym: dict, horizons=(1, 2, 4, 8)) -> list:
    rows = []
    combos = [(90, "le0"), (85, "le0"), (80, "le0"),
              (90, "bottom_third"), (85, "bottom_third"), (80, "bottom_third")]
    for extremity, price_mode in combos:
        masks = {sym: saturation_mask(d, extremity, price_mode) for sym, d in data_by_sym.items()}
        for h in horizons:
            groups_px = [(masks[sym], forward_return(d["rollex_px"], h)) for sym, d in data_by_sym.items()]
            diff, p, ci, n = pooled_block_bootstrap_pvalue(groups_px)
            rows.append(dict(study="S5b", leg="pooled_saturation",
                              bucket=f"pctl>={extremity} & dLong>0 & px={price_mode}", horizon=h,
                              n=n, mean_diff_pct=diff, p=p, ci=ci,
                              note="target=fwd price return, POOLED across KC/CC/SB/CT"))

            groups_mech = [(masks[sym], (d["Spec Long"].shift(-h).values - d["Spec Long"].values) / 1000.0)
                           for sym, d in data_by_sym.items()]
            diff_l, p_l, ci_l, n_l = pooled_block_bootstrap_pvalue(groups_mech)
            rows.append(dict(study="S5b", leg="pooled_saturation_mechanism",
                              bucket=f"pctl>={extremity} & dLong>0 & px={price_mode}", horizon=h,
                              n=n_l, mean_diff_pct=diff_l, p=p_l, ci=ci_l,
                              note="target=fwd chg gross Spec Long (k lots), POOLED"))
    return rows


def pooled_saturation_backtest(data_by_sym: dict, extremity=80, price_mode="bottom_third", hold=4) -> pd.DataFrame:
    """Equal-weight portfolio across commodities: each week, average the
    per-commodity strategy return (0 for commodities not short that week)."""
    per_sym_bt = {}
    for sym, d in data_by_sym.items():
        mask = saturation_mask(d, extremity, price_mode)
        n = len(d)
        pos = np.zeros(n)
        for i in np.where(mask)[0]:
            entry = i + 1
            if entry >= n:
                continue
            pos[entry: min(entry + hold, n)] = -1.0
        px = d["rollex_px"].values
        ret_1w = np.nan_to_num(np.r_[np.nan, px[1:] / px[:-1] - 1.0])
        strat_ret = pos * ret_1w
        cost = np.r_[0, np.abs(np.diff(pos))] * (COST_BPS / 10000.0)
        per_sym_bt[sym] = pd.DataFrame(dict(Date=d["Date"].values, ret=strat_ret - cost,
                                             bh_ret=ret_1w, pos=pos)).rename(
            columns={"ret": f"ret_{sym}", "bh_ret": f"bh_{sym}", "pos": f"pos_{sym}"})

    merged = None
    for bt in per_sym_bt.values():
        merged = bt if merged is None else merged.merge(bt, on="Date", how="inner")

    ret_cols = [c for c in merged.columns if c.startswith("ret_")]
    bh_cols  = [c for c in merged.columns if c.startswith("bh_")]
    pos_cols = [c for c in merged.columns if c.startswith("pos_")]
    return pd.DataFrame(dict(
        Date=merged["Date"],
        ret=merged[ret_cols].mean(axis=1),
        bh_ret=merged[bh_cols].mean(axis=1),
        pos=merged[pos_cols].abs().sum(axis=1).clip(upper=1),  # any commodity short -> "in market"
    ))


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════
def main():
    all_rows = []
    bt_summaries = {}
    data_by_sym = {}
    for sym in ["KC", "CC", "SB", "CT"]:
        d = load_commodity(sym)
        data_by_sym[sym] = d
        print(f"\n{'='*70}\n{sym} - {len(d)} weeks, {d['Date'].min().date()} to {d['Date'].max().date()}\n{'='*70}")

        r1 = study1_extremity(d)
        r2, bt2 = study2_divergence(d)
        r4 = study4_squeeze(d)
        r5, bt5 = study5_saturation(d)
        for r in r1 + r2 + r4 + r5:
            r["symbol"] = sym
        all_rows += r1 + r2 + r4 + r5

        stats2 = backtest_stats(bt2)
        stats5 = backtest_stats(bt5)
        bt_summaries[sym] = dict(divergence=stats2, saturation=stats5)

        print(f"\n  Study 2 backtest ({sym}, long/flat/short on Comm-vs-Spec divergence, "
              f"{COST_BPS}bps cost, signal lagged 1wk):")
        for label, s in stats2.items():
            print(f"    {label:28s}  weeks={s['weeks']:4d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
                  f"b&h_ret={s['bh_total_ret_pct']:+7.1f}%  ann_sharpe={s['ann_sharpe']:+.2f}  "
                  f"max_dd={s['max_dd_pct']:6.1f}%  in_mkt={s['pct_weeks_in_market']:5.1f}%")

        print(f"\n  Study 5 backtest ({sym}, short-only on Spec Saturation Index, "
              f"4wk hold, {COST_BPS}bps cost, signal lagged 1wk):")
        for label, s in stats5.items():
            print(f"    {label:28s}  weeks={s['weeks']:4d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
                  f"b&h_ret={s['bh_total_ret_pct']:+7.1f}%  ann_sharpe={s['ann_sharpe']:+.2f}  "
                  f"max_dd={s['max_dd_pct']:6.1f}%  in_mkt={s['pct_weeks_in_market']:5.1f}%")

    # ── Study 5b: loosened saturation filters, pooled across all 4 commodities ──
    r5b = study5b_pooled_sweep(data_by_sym)
    for r in r5b:
        r["symbol"] = "POOLED(KC+CC+SB+CT)"
    all_rows += r5b

    print(f"\n\n{'='*70}\nSTUDY 5b - Spec Saturation Index, loosened + pooled sweep\n{'='*70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        show5b = pd.DataFrame(r5b)[["leg", "bucket", "horizon", "n", "mean_diff_pct", "p"]].round(3)
        print(show5b.to_string(index=False))

    bt_pooled = pooled_saturation_backtest(data_by_sym, extremity=80, price_mode="bottom_third", hold=4)
    stats_pooled = backtest_stats(bt_pooled)
    print(f"\n  Pooled portfolio backtest (equal-weight KC/CC/SB/CT, short-only, "
          f"pctl>=80 & dLong>0 & px=bottom_third, 4wk hold, {COST_BPS}bps cost, signal lagged 1wk):")
    for label, s in stats_pooled.items():
        print(f"    {label:28s}  weeks={s['weeks']:4d}  strat_ret={s['total_ret_pct']:+7.1f}%  "
              f"b&h_ret={s['bh_total_ret_pct']:+7.1f}%  ann_sharpe={s['ann_sharpe']:+.2f}  "
              f"max_dd={s['max_dd_pct']:6.1f}%  in_mkt={s['pct_weeks_in_market']:5.1f}%")

    df = pd.DataFrame(all_rows)
    df["p_bh"] = bh_correct(df["p"].tolist())

    print(f"\n\n{'='*70}\nBUCKET SIGNIFICANCE TESTS  (mean_diff_pct = bucket mean fwd return "
          f"minus all-other-weeks mean; p from {N_BOOT} moving-block bootstraps, block={BLOCK}wk; "
          f"p_bh = Benjamini-Hochberg corrected across ALL tests below)\n{'='*70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        show = df[["symbol", "study", "leg", "bucket", "horizon", "n",
                   "mean_diff_pct", "p", "p_bh"]].round(3)
        print(show.to_string(index=False))

    sig = df[df["p_bh"] <= 0.10].copy()
    print(f"\n\n{'='*70}\nSURVIVES q<=0.10 after FDR correction: {len(sig)} / {len(df)} tests\n{'='*70}")
    if len(sig):
        print(sig[["symbol", "study", "bucket", "horizon", "mean_diff_pct", "p_bh"]].to_string(index=False))
    else:
        print("None. On this data, none of Study 1 / 2 / 4 / 5 clear a false-discovery-corrected "
              "bar - treat any single-test 'significant' row above as noise until it does.")

    out_path = Path(__file__).parent / "study_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
