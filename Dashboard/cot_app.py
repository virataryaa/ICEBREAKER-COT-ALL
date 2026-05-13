"""
cot_app.py — ICEBREAKER COT Dashboard
Run: streamlit run cot_app.py
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="COMPREHENSIVE COT", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  :root { color-scheme: light !important; }
  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background:#ffffff !important; color:#1a1a1a !important;
  }
  [data-testid="stSidebar"] { background:#f7f8fa !important; }
  [data-testid="stHeader"]  { background:transparent !important; }
  .block-container { padding-top:1.2rem !important; max-width:1600px; }
  div[data-testid="stExpander"] {
    border:1px solid #e0e4ed !important; border-radius:7px !important;
  }
  div[data-testid="stTabs"] button { font-size:0.81rem !important; font-weight:500; }
  hr { border:none !important; border-top:1px solid #e8e8ed !important; margin:.5rem 0 !important; }
  [data-testid="stRadio"] label { font-size:.82rem !important; }
</style>""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_DIR      = Path(__file__).resolve().parent.parent / "Database"
CIT_FILE    = DB_DIR / "cot_cit.parquet"
FO_FILE     = DB_DIR / "cot_disagg_futopt.parquet"
FUT_FILE    = DB_DIR / "cot_disagg_fut.parquet"
ROLLEX_DIR  = DB_DIR / "Rollex"
ROLLEX_MAP  = {
    "KC": "rollex_KC.parquet",
    "CC": "rollex_CC.parquet",
    "CT": "rollex_CT.parquet",
    "SB": "rollex_SB.parquet",
    "RC": "rollex_RC.parquet",
    "LCC":"rollex_LCC.parquet",
}
VAR_LOT_USD = {"KC":375, "CC":10, "SB":1120, "CT":500, "RC":10, "LCC":10}
_CONF_Z     = 2.3263

# ── Commodity config ──────────────────────────────────────────────────────────
COMM_COLORS = {
    "KC":"#1a56db","CC":"#d97706","SB":"#059669",
    "CT":"#7c3aed","RC":"#dc2626","LCC":"#0891b2",
}
COMM_NAMES = {
    "KC":"KC : Arabica Coffee","CC":"CC : NYC Cocoa",
    "SB":"SB : Sugar #11","CT":"CT : Cotton #2",
    "RC":"RC : Robusta Coffee","LCC":"LCC : London Cocoa",
}
CONTRACT_SIZE = {"KC":37500,"CC":10,"SB":112000,"CT":50000,"RC":10,"LCC":10}
CONTRACT_UNIT = {"KC":"lbs","CC":"MT","SB":"lbs","CT":"lbs","RC":"MT","LCC":"MT"}
CIT_COMMS     = {"KC","CC","SB","CT"}

C_LONG  = "#16a34a"
C_SHORT = "#dc2626"
C_NET   = "#1a56db"
C_PRICE = "#f59e0b"
C_OLD   = "#e67e22"
C_NEW   = "#2980b9"
GRAY    = "#6e6e73"

CROP_START_MONTH = 9
_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
CROP_WEEK_TICKS  = {1:"Sep",5:"Oct",9:"Nov",14:"Dec",18:"Jan",22:"Feb",
                    26:"Mar",30:"Apr",35:"May",39:"Jun",43:"Jul",48:"Aug"}
MONTH_TICKS      = {1:"Jan",5:"Feb",9:"Mar",14:"Apr",18:"May",23:"Jun",
                    27:"Jul",32:"Aug",36:"Sep",40:"Oct",45:"Nov",49:"Dec"}

# ── Plot base ─────────────────────────────────────────────────────────────────
_BASE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif",
              color="#1a1a1a", size=11),
)

def _ax(x=False):
    b = dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
             zeroline=True, zerolinecolor="rgba(0,0,0,0.12)", zerolinewidth=1,
             showline=True, linecolor="rgba(0,0,0,0.08)", linewidth=1,
             tickfont=dict(size=10, color="#666"))
    if x:
        b.update(showgrid=False, tickangle=-25)
    return b


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _derive_nets(df, cols):
    pairs = [
        ("Spec Long",     "Spec Short",     "Spec Net"),
        ("Index Long",    "Index Short",    "Index Net"),
        ("Non Rep Long",  "Non Rep Short",  "Non Rep Net"),
        ("Comm Long",     "Comm Short",     "Comm Net"),
        ("MM Long",       "MM Short",       "MM Net"),
        ("Swap Long",     "Swap Short",     "Swap Net"),
        ("Other Long",    "Other Short",    "Other Net"),
        ("Producer Long", "Producer Short", "Comm Net"),
    ]
    for l, s, n in pairs:
        if l in df.columns and s in df.columns and n not in df.columns:
            df[n] = df[l] - df[s]
    return df

def _add_pct(df):
    for col in list(df.columns):
        pct_col = f"Pct OI {col}"
        if (pct_col not in df.columns and "Total OI" in df.columns
                and col not in ("Date","Commodity","Crop","Px")
                and not col.startswith("Traders") and not col.startswith("Conc")
                and not col.startswith("Pct OI")):
            try:
                df[pct_col] = (df[col] / df["Total OI"] * 100).round(2)
            except Exception:
                pass
    return df

@st.cache_data(ttl=600)
def load_cit() -> pd.DataFrame:
    df = pd.read_parquet(CIT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    num = [c for c in df.columns if c not in ("Date","Commodity","Crop")]
    df[num] = df[num].astype(float)
    df = _derive_nets(df, df.columns)
    # Combined spec = Large Spec + Non Rep + Index
    for side in ("Long","Short"):
        df[f"Combined Spec {side}"] = (
            df.get(f"Spec {side}", 0) +
            df.get(f"Non Rep {side}", 0) +
            df.get(f"Index {side}", 0)
        )
    df["Combined Spec Net"] = df["Combined Spec Long"] - df["Combined Spec Short"]
    df = _add_pct(df)
    return df.sort_values(["Commodity","Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_disagg(version: str) -> pd.DataFrame:
    path = FO_FILE if version == "F&O" else FUT_FILE
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    num = [c for c in df.columns if c not in ("Date","Commodity","Crop")]
    df[num] = df[num].astype(float)
    df = _derive_nets(df, df.columns)
    # Combined spec (Disagg) = MM + Other + Non Rep + Swap
    for side in ("Long","Short"):
        df[f"Combined Spec {side}"] = (
            df.get(f"MM {side}", 0) +
            df.get(f"Other {side}", 0) +
            df.get(f"Non Rep {side}", 0) +
            df.get(f"Swap {side}", 0)
        )
    df["Combined Spec Net"] = df["Combined Spec Long"] - df["Combined Spec Short"]
    df = _add_pct(df)
    return df.sort_values(["Commodity","Crop","Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_options_only() -> pd.DataFrame:
    """Options Only = F&O Combined minus Futures Only (numeric cols only)."""
    fo  = pd.read_parquet(FO_FILE)
    fut = pd.read_parquet(FUT_FILE)
    fo["Date"]  = pd.to_datetime(fo["Date"])
    fut["Date"] = pd.to_datetime(fut["Date"])
    id_cols = ["Date","Commodity","Crop"]
    num_fo  = [c for c in fo.columns  if c not in id_cols]
    num_fut = [c for c in fut.columns if c not in id_cols]
    num_both = [c for c in num_fo if c in num_fut]
    merged = fo.merge(fut[id_cols + num_both], on=id_cols, how="left", suffixes=("","_fut"))
    for c in num_both:
        merged[c] = pd.to_numeric(merged[c], errors="coerce") - pd.to_numeric(merged[f"{c}_fut"], errors="coerce")
        merged.drop(columns=[f"{c}_fut"], inplace=True)
    df = merged.copy()
    df = _derive_nets(df, df.columns)
    for side in ("Long","Short"):
        df[f"Combined Spec {side}"] = (
            df.get(f"MM {side}", 0) + df.get(f"Other {side}", 0) +
            df.get(f"Non Rep {side}", 0) + df.get(f"Swap {side}", 0))
    df["Combined Spec Net"] = df["Combined Spec Long"] - df["Combined Spec Short"]
    df = _add_pct(df)
    # Trader counts and concentration % are not valid for options-only
    # (trader overlap between fut/options means subtraction gives wrong counts;
    #  concentration % uses different OI bases so subtracting is meaningless)
    invalid_cols = [c for c in df.columns if
                    c.startswith("Traders") or c.startswith("Conc")]
    df[invalid_cols] = np.nan
    return df.sort_values(["Commodity","Crop","Date"]).reset_index(drop=True)


@st.cache_data(ttl=600)
def load_roll_yield() -> pd.DataFrame:
    path = DB_DIR / "RollYield" / "roll_yield_data.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["Date","Commodity","roll_yield_pct"])
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["roll_yield_pct"] = df["Roll_Yield_1yr"] * 100
    return df[["Date","Commodity","roll_yield_pct"]].sort_values(["Commodity","Date"]).reset_index(drop=True)

@st.cache_data(ttl=600)
def load_rollex(commodity: str) -> pd.DataFrame:
    fname = ROLLEX_MAP.get(commodity)
    if fname is None:
        return pd.DataFrame(columns=["Date","rollex_px","rollex_ret"])
    path = ROLLEX_DIR / fname
    if not path.exists():
        return pd.DataFrame(columns=["Date","rollex_px","rollex_ret"])
    df = pd.read_parquet(path, columns=["rollex_px","rollex_ret"])
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.reset_index()   # index named "Date" → column "Date"
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def _build_var_df(commodity: str) -> pd.DataFrame:
    rx = load_rollex(commodity)
    if rx.empty:
        return pd.DataFrame()
    rx = rx.sort_values("Date").reset_index(drop=True)
    for w in [20, 60, 120]:
        rx[f"vol_{w}"] = rx["rollex_ret"].rolling(w, min_periods=max(5, w // 4)).std()
    return rx.reset_index(drop=True)


def _inject_rollex(cot_df: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """Replace Px with rollex_px via backward merge_asof on Date."""
    rx = load_rollex(commodity)
    if rx.empty:
        return cot_df
    cot_sorted = cot_df.sort_values("Date").copy()
    merged = pd.merge_asof(
        cot_sorted,
        rx[["Date","rollex_px","rollex_ret"]],
        on="Date",
        direction="backward"
    )
    merged["Px"]         = merged["rollex_px"]
    merged["Rollex_ret"] = merged["rollex_ret"]
    merged = merged.drop(columns=["rollex_px","c1","rollex_ret"], errors="ignore")
    return merged.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi_row(items: list, color: str):
    r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 16px'>"
    for lbl, val, sub in items:
        sc = ("#16a34a" if sub and sub.startswith("▲") else
              "#dc2626" if sub and sub.startswith("▼") else "#888")
        sh = (f"<span style='font-size:.63rem;color:{sc};margin-left:4px'>{sub}</span>"
              if sub else "")
        html += (
            f"<div style='background:rgba({r},{g},{b},0.06);"
            f"border:1px solid rgba({r},{g},{b},0.18);border-radius:10px;"
            f"padding:8px 16px;min-width:115px;display:flex;flex-direction:column'>"
            f"<span style='font-size:.56rem;color:#888;text-transform:uppercase;"
            f"letter-spacing:.08em;margin-bottom:3px'>{lbl}</span>"
            f"<span style='font-size:.93rem;font-weight:700;color:{color}'>{val}{sh}</span>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _val(row, col, unit):
    v = row.get(col, np.nan)
    if pd.isna(v): return "—"
    if unit == "k lots":
        return f"{v/1000:.1f}k"
    pc = row.get(f"Pct OI {col}", np.nan)
    return f"{pc:.1f}%" if pd.notna(pc) else f"{v/1000:.1f}k"

def _chg(curr_row, prev_row, col, unit):
    v, p = curr_row.get(col, np.nan), prev_row.get(col, np.nan)
    if pd.isna(v) or pd.isna(p): return ""
    d = (v - p) / 1000
    return f"{'▲' if d>0 else '▼'}{abs(d):.1f}k"

def _px_kpi(curr_row, prev_row):
    v = curr_row.get("Px", np.nan)
    p = prev_row.get("Px", np.nan)
    if pd.isna(v): return "—", ""
    s = f"{v:.2f}"
    chg = "" if pd.isna(p) else (
        f"{'▲' if v>p else '▼'}{abs(v-p):.2f} ({abs((v-p)/p*100):.1f}%)" if p else "")
    return s, chg

def _get_y(d, col, unit):
    if unit == "k lots":
        return d[col] / 1000 if col in d.columns else pd.Series(dtype=float)
    pc = f"Pct OI {col}"
    if pc in d.columns: return d[pc]
    if "Total OI" in d.columns and col in d.columns:
        return (d[col] / d["Total OI"] * 100).round(2)
    return pd.Series(dtype=float)

def show_table(d: pd.DataFrame, pos_cols: list, chg_cols: list, label: str, n=60, scale=True):
    with st.expander(label, expanded=False):
        avail_p = [c for c in pos_cols if c and c in d.columns]
        avail_c = [c for c in chg_cols if c and c in d.columns]
        src = d.sort_values("Date", ascending=False).head(n).copy()
        dates = pd.to_datetime(src["Date"]).dt.strftime("%d %b '%y").tolist()

        # build column data — positions and deltas in k lots (unless scale=False)
        col_data = {}
        for col in avail_p:
            if col == "Px" or not scale:
                col_data[col] = src[col].values
            else:
                col_data[col] = (src[col] / 1000).values
        for col in avail_c:
            if scale:
                col_data[f"Δ {col}"] = (src[col].diff(-1) / 1000).values
            else:
                col_data[f"Δ {col}"] = src[col].diff(-1).values
        if "Px" in avail_p:
            col_data["Px Δ%"] = src["Px"].pct_change(-1).mul(100).values
        if "Total OI" in d.columns and "Total OI" not in avail_p:
            col_data["OI (k)"] = (src["Total OI"] / 1000).values

        signed_cols = {c for c in col_data if c.startswith("Δ") or c == "Px Δ%"}
        px_cols     = {"Px"}
        pct_cols    = {"Px Δ%"}

        def _fmt(col, v):
            if pd.isna(v): return "—"
            if col in pct_cols:    return f"{v:+.2f}%"
            if col in signed_cols: return f"{v:+,.1f}"
            if col in px_cols:     return f"{v:.2f}"
            return f"{v:,.1f}"

        headers = list(col_data.keys())

        hdr_html = "<tr><th class='idx sub'>Date</th>"
        for h in headers:
            lbl = h if not scale or h in ("Px", "Px Δ%", "OI (k)") or h.startswith("Δ") else f"{h} (k)"
            hdr_html += f"<th class='sub'>{lbl}</th>"
        hdr_html += "</tr>"

        body_html = ""
        for i, date in enumerate(dates):
            body_html += f"<tr><td class='idx'>{date}</td>"
            for col in headers:
                v = col_data[col][i]
                txt = _fmt(col, v)
                if col in signed_cols or col in pct_cols:
                    try:
                        fv = float(v)
                        cls = "rpos" if fv > 0 else ("rneg" if fv < 0 else "")
                    except: cls = ""
                else:
                    cls = ""
                body_html += f"<td class='{cls}'>{txt}</td>"
            body_html += "</tr>"

        html = (f"{_RECAP_CSS}<div style='overflow-x:auto;overflow-y:auto;max-height:480px;margin-bottom:6px'>"
                f"<table class='rtbl'><thead>{hdr_html}</thead><tbody>{body_html}</tbody></table></div>")
        st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _add_price(fig, d, secondary_y=True):
    if "Px" not in d.columns or d["Px"].isna().all(): return
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d["Px"], name="Rollex Px",
        line=dict(color=C_PRICE, width=1.2, dash="dot"), opacity=0.65,
        hovertemplate="<b>%{x|%b %Y}</b><br>Rollex Px: %{y:.2f}<extra></extra>",
    ), secondary_y=secondary_y)

def timeseries(d, series, title, ylabel, height=360, price=True):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for s in series:
        fig.add_trace(s["trace"], secondary_y=False)
    if price:
        _add_price(fig, d, secondary_y=True)
    fig.update_layout(
        **_BASE, height=height,
        title=dict(text=title, font=dict(size=12, color="#333"), x=0),
        margin=dict(l=52, r=55, t=42, b=72),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font_size=10, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(x=True), tickformat="%b '%y"),
    )
    fig.update_yaxes(title_text=ylabel, title_font_size=10, secondary_y=False, **_ax())
    fig.update_yaxes(title_text="Rollex Px", title_font_size=10, secondary_y=True,
                     showgrid=False, tickfont=dict(size=10, color=C_PRICE))
    return fig

def bars_weekly(d, col, title, n=13):
    tail = d.tail(n + 1)
    chg  = tail[col].diff().iloc[1:] / 1000
    dates = tail["Date"].iloc[1:]
    fig = go.Figure(go.Bar(
        x=dates, y=chg,
        marker=dict(color=[C_LONG if v >= 0 else C_SHORT for v in chg],
                    opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{x|%d %b %y}</b><br>Δ: %{y:+.1f}k<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.14)")
    fig.update_layout(
        **_BASE, height=290,
        title=dict(text=title, font=dict(size=11, color="#444"), x=0),
        margin=dict(l=50, r=12, t=36, b=68),
        xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
        yaxis=dict(**_ax(), title_text="k lots", title_font_size=10),
        bargap=0.18, showlegend=False,
    )
    return fig

def bars_combined(d, lc, sc, nc, title, color, n=13):
    DARK_GREEN  = "#1a6b1a"
    LIGHT_GREEN = "#7dce7d"
    DARK_RED    = "#8b0000"
    LIGHT_RED   = "#f4a0a0"

    tail  = d.tail(n + 1)
    dates = tail["Date"].iloc[1:]

    if lc in d.columns and sc in d.columns:
        lchg = np.asarray(tail[lc].diff().iloc[1:] / 1000, dtype=float)
        schg = np.asarray(tail[sc].diff().iloc[1:] / 1000, dtype=float)
        long_add    = np.where(lchg > 0,  lchg, 0)
        long_liq    = np.where(lchg < 0,  lchg, 0)
        short_add   = np.where(schg > 0, -schg, 0)   # negated → goes below zero
        short_cover = np.where(schg < 0, -schg, 0)   # negated → goes above zero
    else:
        long_add = long_liq = short_add = short_cover = None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if long_add is not None:
        for arr, name, clr in [
            (long_add,    "Long Add",    DARK_GREEN),
            (long_liq,    "Long Liq.",   LIGHT_GREEN),
            (short_add,   "Short Add",   DARK_RED),
            (short_cover, "Short Cover", LIGHT_RED),
        ]:
            fig.add_trace(go.Bar(x=dates, y=arr, name=name,
                marker_color=clr, opacity=0.92,
                hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>{name}: %{{y:+.2f}}k<extra></extra>"),
                secondary_y=False)

    if "Px" in d.columns:
        px_vals = np.asarray(tail["Px"].iloc[1:], dtype=float)
        fig.add_trace(go.Scatter(x=dates, y=px_vals, name="Rollex Px", mode="lines",
            line=dict(color=C_PRICE, width=1.8),
            hovertemplate="<b>%{x|%d %b %y}</b><br>Rollex Px: %{y:.2f}<extra></extra>"),
            secondary_y=True)

    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.14)")
    fig.update_layout(**_BASE, height=340, barmode="relative",
        title=dict(text=title, font=dict(size=11, color="#444"), x=0),
        margin=dict(l=50, r=55, t=36, b=72),
        legend=dict(orientation="h", y=-0.26, x=0.5, xanchor="center", font_size=10),
        xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
        bargap=0.12)
    fig.update_yaxes(title_text="k lots", title_font_size=10, secondary_y=False, **_ax())
    fig.update_yaxes(title_text="Rollex Px", title_font_size=10, secondary_y=True,
                     showgrid=False, tickfont=dict(size=10, color=C_PRICE))
    return fig


def histogram_dist(d, col, color, title):
    chg = (d[col] / 1000).diff().dropna()
    if chg.empty: return go.Figure().update_layout(**_BASE, height=260)
    lv = float(chg.iloc[-1])
    r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    fig = go.Figure(go.Histogram(
        x=chg, nbinsx=30, showlegend=False,
        marker=dict(color=f"rgba({r},{g},{b},0.70)", line=dict(color="white", width=0.5)),
        hovertemplate="Δ: %{x:.1f}k<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=lv, line_dash="dash", line_color=C_SHORT, line_width=1.8,
                  annotation_text=f" {lv:+.1f}k", annotation_font_size=9,
                  annotation_font_color=C_SHORT)
    fig.update_layout(
        **_BASE, height=260,
        title=dict(text=f"Weekly Δ dist — {title}", font=dict(size=11, color="#444"), x=0),
        margin=dict(l=40, r=12, t=36, b=36),
        xaxis=dict(**_ax(x=True), title_text="k lots"),
        yaxis=dict(**_ax()), showlegend=False,
    )
    return fig

def seasonal(d, col, color, title):
    if d.empty or col not in d.columns:
        return go.Figure().update_layout(**_BASE, height=340,
            title=dict(text=f"Seasonality — {title}  (no data)", font=dict(size=11,color="#999"), x=0))
    s = d[["Date", col]].copy()
    s["v"]    = s[col] / 1000
    s["Week"] = s["Date"].dt.isocalendar().week.astype(int)
    s["Year"] = s["Date"].dt.year.astype(int)
    pivot = s.pivot_table(index="Week", columns="Year", values="v", aggfunc="mean")
    pivot = pivot[pivot.index <= 52]
    cur_year  = int(s["Year"].max())
    hist      = pivot[[c for c in pivot.columns if int(c) < cur_year]]
    if hist.empty or hist.shape[1] == 0:
        # Only current-year data — just show current year line, no band
        fig = go.Figure()
        if cur_year in pivot.columns:
            cy = pivot[cur_year].dropna()
            fig.add_trace(go.Scatter(x=cy.index, y=cy.values, mode="lines+markers",
                name=str(cur_year), line=dict(color=color, width=2.5),
                marker=dict(size=4.5, color=color)))
        fig.update_layout(**_BASE, height=340,
            title=dict(text=f"Seasonality — {title}  ·  k lots  (current year only)",
                       font=dict(size=11,color="#999"), x=0),
            margin=dict(l=50,r=20,t=42,b=60),
            xaxis=dict(**_ax(), title_text="Week"),
            yaxis=dict(**_ax(), title_text="k lots", title_font_size=10))
        return fig
    p25, p75, med = hist.quantile(0.25,axis=1), hist.quantile(0.75,axis=1), hist.median(axis=1)
    r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    fig = go.Figure()
    for yr in hist.columns:
        fig.add_trace(go.Scatter(x=pivot.index, y=hist[yr], mode="lines",
            line=dict(color="rgba(160,160,160,0.14)", width=1),
            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=list(p75.index)+list(p75.index[::-1]),
        y=list(p75.values)+list(p25.values[::-1]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.10)",
        line=dict(width=0), name="25–75th pct", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=med.index, y=med.values, mode="lines", name="Median",
        line=dict(color=f"rgba({r},{g},{b},0.5)", width=1.6, dash="dash"),
        hovertemplate="Wk %{x}  Median: %{y:.1f}k<extra></extra>"))
    if cur_year in pivot.columns:
        cy = pivot[cur_year].dropna()
        fig.add_trace(go.Scatter(x=cy.index, y=cy.values, mode="lines+markers",
            name=str(cur_year), line=dict(color=color, width=2.5),
            marker=dict(size=4.5, color=color),
            hovertemplate=f"Wk %{{x}}  {cur_year}: %{{y:.1f}}k<extra></extra>"))

    MTICKS = {1:"Jan",5:"Feb",9:"Mar",14:"Apr",18:"May",23:"Jun",
              27:"Jul",32:"Aug",36:"Sep",40:"Oct",45:"Nov",49:"Dec"}
    fig.update_layout(
        **_BASE, height=340,
        title=dict(text=f"Seasonality — {title}  ·  k lots", font=dict(size=12,color="#333"), x=0),
        margin=dict(l=50,r=20,t=42,b=60),
        legend=dict(orientation="h",y=-0.18,x=0.5,xanchor="center",font_size=10),
        xaxis=dict(**_ax(), tickmode="array",
                   tickvals=list(MTICKS.keys()), ticktext=list(MTICKS.values()),
                   title_text="Week"),
        yaxis=dict(**_ax(), title_text="k lots", title_font_size=10),
    )
    return fig

def scatter_2d(d, x_col, y_col, color, title, xlabel, ylabel):
    xraw = d[x_col].pct_change()*100 if x_col=="Px" else d[x_col].diff()/1000
    x = np.asarray(xraw, dtype=float)
    y = np.asarray(d[y_col].diff()/1000, dtype=float)
    dates = np.asarray(d["Date"])
    mask = ~(np.isnan(x)|np.isnan(y))
    x,y,dates = x[mask],y[mask],dates[mask]
    if len(x)<5: return go.Figure().update_layout(**_BASE, height=340)
    r2 = float(np.corrcoef(x,y)[0,1]**2)
    sl,ic = np.polyfit(x,y,1)
    xl = np.linspace(x.min(),x.max(),200)
    rec = (dates-dates.min()).astype("timedelta64[D]").astype(float)
    nr  = rec/max(rec.max(),1)
    rv,gv,bv = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
        marker=dict(color=nr,
            colorscale=[[0,"rgba(200,210,230,0.5)"],[1,f"rgba({rv},{gv},{bv},0.85)"]],
            size=7, line=dict(width=0.5, color="white")),
        text=pd.to_datetime(dates).strftime("%Y-%m-%d"),
        hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        showlegend=False))
    fig.add_trace(go.Scatter(x=xl, y=sl*xl+ic, mode="lines",
        line=dict(color=color, width=1.6, dash="dash"), showlegend=False))
    fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", showlegend=False,
        marker=dict(symbol="star", size=14, color=C_SHORT,
                    line=dict(width=1.2, color="white"))))
    fig.update_layout(
        **_BASE, height=340,
        title=dict(text=f"{title}   <span style='font-size:10px;color:#888'>R²={r2:.2f}</span>",
                   font=dict(size=12,color="#333"), x=0),
        margin=dict(l=52,r=20,t=48,b=48),
        xaxis=dict(**_ax(x=True), title_text=xlabel),
        yaxis=dict(**_ax(), title_text=ylabel),
    )
    return fig


def scatter_3d(x, y, z, dates, color, title, xlabel, ylabel, zlabel, height=540):
    x, y, z = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    dates    = np.asarray(dates, dtype="datetime64[ns]")
    mask     = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z, dates = x[mask], y[mask], z[mask], dates[mask]
    if len(x) < 5:
        return go.Figure().update_layout(height=height,
            title=dict(text=title + "  [insufficient data]", font_size=12))
    corr_xy = float(np.corrcoef(x, y)[0, 1])
    corr_xz = float(np.corrcoef(x, z)[0, 1])
    corr_yz = float(np.corrcoef(y, z)[0, 1])
    rec  = (dates - dates.min()).astype("timedelta64[D]").astype(float)
    nr   = rec / max(rec.max(), 1)
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    def _ax3(label, bg):
        return dict(title=dict(text=label, font=dict(size=10, color="#555")),
                    tickfont=dict(size=9, color="#777"),
                    gridcolor="rgba(0,0,0,0.10)", gridwidth=1,
                    showbackground=True, backgroundcolor=bg,
                    zerolinecolor="rgba(0,0,0,0.20)", zerolinewidth=2)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers",
        marker=dict(color=nr,
            colorscale=[[0,"rgba(200,210,230,0.45)"],[1,f"rgba({r},{g},{b},0.9)"]],
            size=5, line=dict(width=0.6, color="white")),
        text=pd.to_datetime(dates).strftime("%Y-%m-%d"),
        hovertemplate=(f"<b>%{{text}}</b><br>{xlabel}: %{{x:.2f}}<br>"
                       f"{ylabel}: %{{y:.2f}}<br>{zlabel}: %{{z:.2f}}<extra></extra>"),
        showlegend=False))
    fig.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode="markers",
        showlegend=False,
        marker=dict(symbol="diamond", size=10, color=C_SHORT,
                    line=dict(width=1.5, color="white")),
        hovertemplate=(f"<b>Latest</b><br>{xlabel}: {x[-1]:.2f}<br>"
                       f"{ylabel}: {y[-1]:.2f}<br>{zlabel}: {z[-1]:.2f}<extra></extra>")))
    corr_str = f"r(X,Y)={corr_xy:+.2f}  ·  r(X,Z)={corr_xz:+.2f}  ·  r(Y,Z)={corr_yz:+.2f}"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system,BlinkMacSystemFont,sans-serif", color="#2d2d2d", size=11),
        title=dict(text=f"{title}<br><span style='font-size:10px;color:#888'>{corr_str}</span>",
                   font=dict(size=12, color="#444"), x=0),
        height=height, margin=dict(l=0, r=0, t=70, b=20),
        scene=dict(
            aspectmode="cube",
            camera=dict(up=dict(x=0,y=0,z=1), center=dict(x=0,y=0,z=-0.1),
                        eye=dict(x=1.6,y=1.6,z=0.8)),
            xaxis=_ax3(xlabel, "rgba(240,244,255,0.6)"),
            yaxis=_ax3(ylabel, "rgba(240,255,244,0.6)"),
            zaxis=_ax3(zlabel, "rgba(255,248,240,0.6)"),
        ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SPEC
# ══════════════════════════════════════════════════════════════════════════════
CIT_SPEC = {
    "Large Spec":    {"long":"Spec Long",    "short":"Spec Short",    "net":"Spec Net",    "spread":None},
    "Non-Rep":       {"long":"Non Rep Long", "short":"Non Rep Short", "net":"Non Rep Net", "spread":None},
    "Index Traders": {"long":"Index Long",   "short":"Index Short",   "net":"Index Net",   "spread":None},
    "Large Spec + Index + Non-Rep": {"long":"Combined Spec Long","short":"Combined Spec Short","net":"Combined Spec Net","spread":None},
}
DISAGG_SPEC = {
    "Managed Money":              {"long":"MM Long",    "short":"MM Short",    "net":"MM Net",    "spread":"MM Spread"},
    "Other Rept":                 {"long":"Other Long", "short":"Other Short", "net":"Other Net", "spread":"Other Spread"},
    "Non-Rep":                    {"long":"Non Rep Long","short":"Non Rep Short","net":"Non Rep Net","spread":None},
    "Swap Dealers":               {"long":"Swap Long",  "short":"Swap Short",  "net":"Swap Net",  "spread":"Swap Spread"},
    "MM + Other + Non-Rep + Swap":{"long":"Combined Spec Long","short":"Combined Spec Short","net":"Combined Spec Net","spread":None},
}

def render_spec(d, report, color):
    cats = CIT_SPEC if report == "CIT" else DISAGG_SPEC
    cat  = st.selectbox("Category", list(cats.keys()), key="spec_cat")
    unit = "k lots"

    cfg = cats[cat]
    lc, sc, nc, spc = cfg["long"], cfg["short"], cfg["net"], cfg["spread"]
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    latest = d.iloc[-1]; prev = d.iloc[-2] if len(d)>1 else d.iloc[-1]
    pxv, pxchg = _px_kpi(latest, prev)

    kpi_items = [
        ("Long",  _val(latest, lc, unit),       _chg(latest, prev, lc, unit)),
        ("Short", _val(latest, sc, unit),       _chg(latest, prev, sc, unit)),
        ("Net",   _val(latest, nc, "k lots"),   _chg(latest, prev, nc, "k lots")),
        ("Total OI", f"{latest.get('Total OI',np.nan)/1000:.1f}k"
                     if pd.notna(latest.get('Total OI')) else "—", ""),
        ("Rollex Px", pxv, pxchg),
    ]
    if spc and spc in d.columns:
        kpi_items.insert(3, ("Spread", _val(latest, spc, unit), _chg(latest, prev, spc, unit)))
    kpi_row(kpi_items, color)

    # Combined timeseries — Long, Short, Net all in selected unit + Price secondary
    ylabel = "k lots" if unit == "k lots" else "% of OI"
    suffix = "k" if unit == "k lots" else "%"
    traces = []
    if lc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, lc, unit), name="Long",
            line=dict(color=C_LONG, width=2.0),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Long: %{{y:.1f}}{suffix}<extra></extra>")})
    if sc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, sc, unit), name="Short",
            line=dict(color=C_SHORT, width=2.0),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Short: %{{y:.1f}}{suffix}<extra></extra>")})
    if nc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, nc, unit), name="Net",
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.09)",
            line=dict(color=color, width=2.2),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Net: %{{y:.1f}}{suffix}<extra></extra>")})
    if spc and spc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, spc, unit), name="Spread",
            line=dict(color="#94a3b8", width=1.4, dash="dot"),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Spread: %{{y:.1f}}{suffix}<extra></extra>")})
    st.plotly_chart(timeseries(d, traces, f"{cat}  ·  {ylabel}", ylabel), width='stretch')

    # Stacked Long Add/Liq + Short Add/Cover bars + Price
    st.plotly_chart(bars_combined(d, lc, sc, nc, f"{cat} — weekly flow  ·  k lots", color),
                    width='stretch')

    with st.expander("Seasonality", expanded=False):
        seas_items = [(lc, C_LONG, "Long"), (sc, C_SHORT, "Short"), (nc, color, "Net")]
        avail_s = [(col, clr, lbl) for col, clr, lbl in seas_items if col in d.columns]
        scols = st.columns(len(avail_s))
        for ch, (col, clr, lbl) in zip(scols, avail_s):
            with ch:
                st.plotly_chart(seasonal(d, col, clr, f"{cat} {lbl}"), width='stretch')

    show_table(d, [lc, sc, nc] + ([spc] if spc else []) + ["Px"],
               [lc, sc, nc], f"Data table — {cat}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMMERCIAL
# ══════════════════════════════════════════════════════════════════════════════
def render_commercial(d, report, color):
    is_disagg = report == "Disagg"
    lc  = "Producer Long"  if is_disagg else "Comm Long"
    sc  = "Producer Short" if is_disagg else "Comm Short"
    nc  = "Comm Net"
    lbl = "Producer/Merchant" if is_disagg else "Commercial"
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    unit = "k lots"
    latest = d.iloc[-1]; prev = d.iloc[-2] if len(d)>1 else d.iloc[-1]
    pxv, pxchg = _px_kpi(latest, prev)

    kpi_items = [
        ("Long",  _val(latest, lc, unit),       _chg(latest, prev, lc, unit)),
        ("Short", _val(latest, sc, unit),       _chg(latest, prev, sc, unit)),
        ("Net",   _val(latest, nc, "k lots"),   _chg(latest, prev, nc, "k lots")),
        ("Total OI", f"{latest.get('Total OI',np.nan)/1000:.1f}k"
                     if pd.notna(latest.get('Total OI')) else "—", ""),
        ("Rollex Px", pxv, pxchg),
    ]
    kpi_row(kpi_items, color)

    # Combined timeseries — Long, Short, Net in selected unit + Price secondary
    ylabel = "k lots" if unit == "k lots" else "% of OI"
    suffix = "k" if unit == "k lots" else "%"
    traces = []
    for col, name, clr in [(lc, "Long", C_LONG), (sc, "Short", C_SHORT)]:
        if col in d.columns:
            traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, col, unit), name=name,
                line=dict(color=clr, width=2.0),
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{name}: %{{y:.1f}}{suffix}<extra></extra>")})
    if nc in d.columns:
        traces.append({"trace": go.Scatter(x=d["Date"], y=_get_y(d, nc, unit), name="Net",
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.07)",
            line=dict(color=color, width=2.2),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Net: %{{y:.1f}}{suffix}<extra></extra>")})
    st.plotly_chart(timeseries(d, traces, f"{lbl}  ·  {ylabel}", ylabel), width='stretch')

    # Stacked Long Add/Liq + Short Add/Cover bars + Price
    st.plotly_chart(bars_combined(d, lc, sc, nc, f"{lbl} — weekly flow  ·  k lots", color),
                    width='stretch')

    with st.expander("Seasonality", expanded=False):
        seas = [(lc, C_LONG, "Long"), (sc, C_SHORT, "Short"), (nc, color, "Net")]
        avail_s = [(col, clr, name) for col, clr, name in seas if col in d.columns]
        scols = st.columns(len(avail_s))
        for ch, (col, clr, name) in zip(scols, avail_s):
            with ch:
                st.plotly_chart(seasonal(d, col, clr, f"{lbl} {name}"), width='stretch')

    show_table(d, [lc, sc, nc, "Px"], [lc, sc, nc], f"Data table — {lbl}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SPREADING (Disagg only)
# ══════════════════════════════════════════════════════════════════════════════
SPREAD_COLS = {
    "Managed Money": ("MM Spread",    C_NET),
    "Swap Dealers":  ("Swap Spread",  "#7c3aed"),
    "Other Rept":    ("Other Spread", "#d97706"),
}

def render_spreading(d, color):
    st.markdown(
        "<p style='font-size:.78rem;color:#666;margin-bottom:8px'>"
        "Spreading = offsetting long/short positions in different delivery months. "
        "Shown per category in lots and % of OI.</p>", unsafe_allow_html=True)

    unit = st.radio("Unit", ["k lots","% of OI"], horizontal=True, key="spread_unit")
    ylabel = "k lots" if unit=="k lots" else "% of OI"

    latest = d.iloc[-1]
    kpi_items = []
    for lbl,(col,clr) in SPREAD_COLS.items():
        if col in d.columns:
            kpi_items.append((lbl, _val(latest,col,unit), ""))
    kpi_row(kpi_items, color)

    # All spreads on one timeseries
    traces = []
    for lbl,(col,clr) in SPREAD_COLS.items():
        if col not in d.columns: continue
        y = _get_y(d,col,unit)
        traces.append({"trace": go.Scatter(x=d["Date"],y=y,name=lbl,
            line=dict(color=clr,width=2.0),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{lbl}: %{{y:.1f}}<extra></extra>")})
    st.plotly_chart(timeseries(d,traces,f"Spreading by Category  ·  {ylabel}",ylabel), width='stretch')

    # Weekly change per category
    avail = [(lbl,col,clr) for lbl,(col,clr) in SPREAD_COLS.items() if col in d.columns]
    cols  = st.columns(len(avail))
    for i,(lbl,col,clr) in enumerate(avail):
        with cols[i]:
            st.plotly_chart(bars_weekly(d,col,f"{lbl} Spread — weekly Δ"), width='stretch')

    with st.expander("Seasonality", expanded=False):
        ch1,ch2,ch3 = st.columns(3)
        for ch,(lbl,col,clr) in zip([ch1,ch2,ch3], avail):
            with ch: st.plotly_chart(seasonal(d,col,clr,f"{lbl} Spread"), width='stretch')

    show_table(d, [col for _,(col,_) in SPREAD_COLS.items() if col in d.columns] + ["Total OI"],
               [col for _,(col,_) in SPREAD_COLS.items() if col in d.columns],
               "Data table — Spreading")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OLD / NEW CROP (Disagg only)
# ══════════════════════════════════════════════════════════════════════════════
# ── Old/New helpers ───────────────────────────────────────────────────────────
def _crop_year_label(dt, sm=CROP_START_MONTH):
    y, m = dt.year, dt.month
    return f"{str(y)[2:]}/{str(y+1)[2:]}" if m >= sm else f"{str(y-1)[2:]}/{str(y)[2:]}"

def _crop_week_num(dt, sm=CROP_START_MONTH):
    y = dt.year if dt.month >= sm else dt.year - 1
    return max(1, (dt - pd.Timestamp(y, sm, 1)).days // 7 + 1)

def _current_crop_year_label(sm=CROP_START_MONTH):
    return _crop_year_label(pd.Timestamp.now(), sm)

def _on_seasonal_wide(d_crops):
    old   = d_crops[d_crops["Crop"]=="Old"].set_index("Date").sort_index()
    other = d_crops[d_crops["Crop"]=="Other"].set_index("Date").sort_index()
    common = old.index.intersection(other.index)
    if common.empty: return pd.DataFrame()
    old, other = old.loc[common], other.loc[common]
    oi_sum = old["Total OI"] + other["Total OI"]
    wide = pd.DataFrame({
        "OI Old %":       old["Total OI"] / oi_sum * 100,
        "MM Net Old":     old["MM Net"],     "MM Net New":   other["MM Net"],
        "MM Long Old":    old["MM Long"],    "MM Long New":  other["MM Long"],
        "MM Short Old":   old["MM Short"],   "MM Short New": other["MM Short"],
        "MM Diff":        old["MM Net"] - other["MM Net"],
        "Comm Net Old":   old["Comm Net"],   "Comm Net New": other["Comm Net"],
        "Comm Long Old":  old["Producer Long"]  if "Producer Long"  in old.columns else pd.Series(dtype=float),
        "Comm Long New":  other["Producer Long"] if "Producer Long"  in other.columns else pd.Series(dtype=float),
        "Comm Short Old": old["Producer Short"] if "Producer Short" in old.columns else pd.Series(dtype=float),
        "Comm Short New": other["Producer Short"] if "Producer Short" in other.columns else pd.Series(dtype=float),
        "Week": pd.Series(common.isocalendar().week.astype(int).values, index=common),
        "Year": pd.Series(common.year, index=common),
    }, index=common)
    return wide.reset_index()

def _seas_chart(wide, metric, title, accent, ylabel="k lots", by_week=True, sm=CROP_START_MONTH):
    if wide.empty or metric not in wide.columns:
        return go.Figure().update_layout(**_BASE, height=340)
    grp = "Week" if by_week else "CropWeek"
    yr_col = "Year" if by_week else "CropYear"
    pivot = wide.pivot_table(index=grp, columns=yr_col, values=metric, aggfunc="mean")
    pivot = pivot[pivot.index <= 52]
    if by_week:
        cur = int(wide["Year"].max())
        hist_cols = [c for c in pivot.columns if c < cur]
    else:
        cur = _current_crop_year_label(sm)
        hist_cols = [c for c in pivot.columns if c != cur]
    hist = pivot[hist_cols] if hist_cols else pivot
    p25, p75, med = hist.quantile(0.25, axis=1), hist.quantile(0.75, axis=1), hist.median(axis=1)
    r, g, b = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
    fig = go.Figure()
    for yr in hist_cols:
        fig.add_trace(go.Scatter(x=pivot.index, y=hist[yr], mode="lines",
            line=dict(color="rgba(150,150,150,0.18)", width=1), showlegend=False, hoverinfo="skip"))
    xs = list(p75.index) + list(p75.index[::-1])
    fig.add_trace(go.Scatter(x=xs, y=list(p75.values)+list(p25.values[::-1]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.10)",
        line=dict(width=0), name="25–75th pct", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=med.index, y=med.values, mode="lines", name="Median",
        line=dict(color=f"rgba({r},{g},{b},0.55)", width=1.6, dash="dash")))
    if cur in pivot.columns:
        cy = pivot[cur].dropna()
        fig.add_trace(go.Scatter(x=cy.index, y=cy.values, mode="lines+markers",
            name=str(cur), line=dict(color=C_OLD, width=2.6), marker=dict(size=5, color=C_OLD)))
    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.12)")
    if by_week:
        xticks = dict(tickvals=list(MONTH_TICKS.keys()), ticktext=list(MONTH_TICKS.values()))
    else:
        xticks = dict(tickvals=list(CROP_WEEK_TICKS.keys()),
                      ticktext=[_MONTHS[(sm - 1 + i) % 12] for i in range(12)])
    fig.update_layout(**_BASE, height=340,
        title=dict(text=title, font=dict(size=12, color="#444"), x=0),
        margin=dict(l=50, r=20, t=40, b=60),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font_size=10, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(x=True), title_text="", **xticks),
        yaxis=dict(**_ax(), title_text=ylabel, title_font_size=10))
    return fig



def _seas_chart_overlay(wide, metric_old, metric_new, title, ylabel, sm=CROP_START_MONTH):
    """Seasonality chart overlaying Old Crop and New Crop on the same axes."""
    if wide.empty:
        return go.Figure().update_layout(**_BASE, height=360)
    cur = _current_crop_year_label(sm)

    def _pivot(metric):
        if metric not in wide.columns:
            return None
        pv = wide.pivot_table(index="CropWeek", columns="CropYear", values=metric, aggfunc="mean")
        return pv[pv.index <= 52]

    pv_old = _pivot(metric_old)
    pv_new = _pivot(metric_new)

    fig = go.Figure()

    for pv, clr_hist, clr_cur, clr_band, label in [
        (pv_old, "rgba(230,126,34,0.15)", C_OLD, "rgba(230,126,34,0.12)", "Old Crop"),
        (pv_new, "rgba(41,128,185,0.15)",  C_NEW, "rgba(41,128,185,0.12)",  "New Crop"),
    ]:
        if pv is None:
            continue
        hist_cols = [c2 for c2 in pv.columns if c2 != cur]
        hist = pv[hist_cols] if hist_cols else pv

        # faint historical lines
        for yr in hist_cols:
            fig.add_trace(go.Scatter(
                x=pv.index, y=hist[yr], mode="lines",
                line=dict(color=clr_hist, width=1),
                showlegend=False, hoverinfo="skip"
            ))

        # IQR band
        if len(hist_cols) >= 4:
            p25 = hist.quantile(0.25, axis=1)
            p75 = hist.quantile(0.75, axis=1)
            xs  = list(p75.index) + list(p75.index[::-1])
            fig.add_trace(go.Scatter(
                x=xs, y=list(p75.values) + list(p25.values[::-1]),
                fill="toself", fillcolor=clr_band,
                line=dict(width=0), name=f"{label} 25-75%", hoverinfo="skip"
            ))

        # current crop year — bold
        if cur in pv.columns:
            cy = pv[cur].dropna()
            fig.add_trace(go.Scatter(
                x=cy.index, y=cy.values, mode="lines+markers",
                name=f"{label} {cur}",
                line=dict(color=clr_cur, width=2.6),
                marker=dict(size=4, color=clr_cur),
                hovertemplate=f"<b>{label} {cur}</b><br>Week %{{x}}: %{{y:.1f}}<extra></extra>"
            ))

    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.12)")
    xticks = dict(
        tickvals=list(CROP_WEEK_TICKS.keys()),
        ticktext=[_MONTHS[(sm - 1 + i) % 12] for i in range(12)]
    )
    fig.update_layout(
        **_BASE, height=360,
        title=dict(text=title, font=dict(size=12, color="#444"), x=0),
        margin=dict(l=50, r=20, t=40, b=70),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font_size=10, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(x=True), **xticks),
        yaxis=dict(**_ax(), title_text=ylabel, title_font_size=10),
    )
    return fig


def render_old_new(d_crops, color):
    old   = d_crops[d_crops["Crop"]=="Old"].set_index("Date").sort_index()
    other = d_crops[d_crops["Crop"]=="Other"].set_index("Date").sort_index()
    alla  = d_crops[d_crops["Crop"]=="All"].set_index("Date").sort_index()

    if old.empty and other.empty:
        st.info("No Old/Other crop data in selected range."); return

    other_check = d_crops[d_crops["Crop"]=="Other"]["Total OI"].dropna()
    if other_check.empty:
        st.info("Old/New crop split not available for this commodity."); return

    with st.expander("Data table  ·  Old Crop / New Crop", expanded=False):
        common_dates = old.index.union(other.index).sort_values()[::-1][:30]
        common_dates_ext = old.index.union(other.index).sort_values()[::-1][:31]

        def _get_s(df, col, dates):
            return (df.reindex(dates)[col] / 1000) if col in df.columns else pd.Series(np.nan, index=dates)

        def _build_tbl(dates):
            data = {}
            for src, lbl in [("MM Net","MM Net Old"),("Comm Net","Comm Net Old")]:
                data[("Net · k lots", lbl)] = _get_s(old, src, dates).values
            for src, lbl in [("MM Net","MM Net New"),("Comm Net","Comm Net New")]:
                data[("Net · k lots", lbl)] = _get_s(other, src, dates).values
            for src, lbl in [("MM Long","Old"),("MM Short","Old"),("Producer Long","Old"),("Producer Short","Old")]:
                data[(src.replace("Producer","Prod"), lbl)] = _get_s(old, src, dates).values
            for src, lbl in [("MM Long","New"),("MM Short","New"),("Producer Long","New"),("Producer Short","New")]:
                data[(src.replace("Producer","Prod"), lbl)] = _get_s(other, src, dates).values
            data[("OI · k lots", "Old")] = _get_s(old, "Total OI", dates).values
            data[("OI · k lots", "New")] = _get_s(other, "Total OI", dates).values
            return pd.DataFrame(data, index=dates)

        tbl_df = _build_tbl(common_dates)
        tbl_df.index = pd.to_datetime(tbl_df.index).strftime("%d %b '%y")
        tbl_df.index.name = None
        st.markdown(_recap_html(tbl_df, scroll=True), unsafe_allow_html=True)

        tbl_ext = _build_tbl(common_dates_ext)
        chg_df = tbl_ext.diff(-1).iloc[:len(common_dates)]
        chg_df.index = pd.to_datetime(common_dates).strftime("%d %b '%y")
        chg_df.index.name = None
        st.markdown(
            f"<p style='font-size:.72rem;color:{GRAY};margin:8px 0 2px'>Weekly change  ·  k lots</p>",
            unsafe_allow_html=True)
        all_groups = {g for g, _ in chg_df.columns}
        st.markdown(_recap_html(chg_df, signed_groups=all_groups, scroll=True), unsafe_allow_html=True)

    with st.expander("Seasonality  ·  Old vs New Crop (adjustable start)", expanded=False):
        wide_full = _on_seasonal_wide(d_crops)
        if not wide_full.empty:
            sm = st.selectbox("Crop year starts in", list(range(1,13)),
                              index=CROP_START_MONTH-1, format_func=lambda m: _MONTHS[m-1],
                              key="cy_start_on")
            wide_cy = wide_full.copy()
            wide_cy["CropYear"] = pd.to_datetime(wide_cy["Date"]).apply(lambda d: _crop_year_label(d, sm))
            wide_cy["CropWeek"] = pd.to_datetime(wide_cy["Date"]).apply(lambda d: _crop_week_num(d, sm))
            cur_cy = _current_crop_year_label(sm)
            st.markdown(
                f"<p style='font-size:.75rem;color:{GRAY};margin-bottom:4px'>"
                f"Each line = one crop year · Bold = current ({cur_cy}) · Shaded = 25–75th pct</p>",
                unsafe_allow_html=True)

            def _sc(metric, title, ylabel="k lots"):
                return _seas_chart(wide_cy, metric, title, color, ylabel, by_week=False, sm=sm)

            # ── Managed Money ─────────────────────────────────────────────────
            st.markdown("<div style='font-size:.75rem;font-weight:700;color:#374151;"
                        "margin:14px 0 6px;letter-spacing:.04em'>MANAGED MONEY</div>",
                        unsafe_allow_html=True)
            mm1, mm2, mm3 = st.columns(3)
            with mm1: st.plotly_chart(_sc("MM Net Old",   "MM Net (Old)  ·  k lots"),   width='stretch')
            with mm2: st.plotly_chart(_sc("MM Net New",   "MM Net (New)  ·  k lots"),   width='stretch')
            with mm3: st.plotly_chart(_sc("MM Diff",      "MM Net (Old − New)  ·  k lots"),    width='stretch')
            mm4, mm5, _ = st.columns(3)
            with mm4: st.plotly_chart(_sc("MM Long Old",  "MM Long (Old)  ·  k lots"),  width='stretch')
            with mm5: st.plotly_chart(_sc("MM Long New",  "MM Long (New)  ·  k lots"),  width='stretch')
            mm6, mm7, _ = st.columns(3)
            with mm6: st.plotly_chart(_sc("MM Short Old", "MM Short (Old)  ·  k lots"), width='stretch')
            with mm7: st.plotly_chart(_sc("MM Short New", "MM Short (New)  ·  k lots"), width='stretch')

            # ── Commercial ────────────────────────────────────────────────────
            st.markdown("<div style='font-size:.75rem;font-weight:700;color:#374151;"
                        "margin:14px 0 6px;letter-spacing:.04em'>COMMERCIAL</div>",
                        unsafe_allow_html=True)
            cm1, cm2, _ = st.columns(3)
            with cm1: st.plotly_chart(_sc("Comm Net Old",   "Comm Net (Old)  ·  k lots"),   width='stretch')
            with cm2: st.plotly_chart(_sc("Comm Net New",   "Comm Net (New)  ·  k lots"),   width='stretch')
            cm3, cm4, _ = st.columns(3)
            with cm3: st.plotly_chart(_sc("Comm Long Old",  "Comm Long (Old)  ·  k lots"),  width='stretch')
            with cm4: st.plotly_chart(_sc("Comm Long New",  "Comm Long (New)  ·  k lots"),  width='stretch')
            cm5, cm6, _ = st.columns(3)
            with cm5: st.plotly_chart(_sc("Comm Short Old", "Comm Short (Old)  ·  k lots"), width='stretch')
            with cm6: st.plotly_chart(_sc("Comm Short New", "Comm Short (New)  ·  k lots"), width='stretch')

            # ── Open Interest ─────────────────────────────────────────────────
            st.markdown("<div style='font-size:.75rem;font-weight:700;color:#374151;"
                        "margin:14px 0 6px;letter-spacing:.04em'>OPEN INTEREST</div>",
                        unsafe_allow_html=True)
            oi1, oi2, _ = st.columns(3)
            with oi1: st.plotly_chart(_sc("OI Old %", "OI % (Old)", "%"), width='stretch')

    # ── OI split ──────────────────────────────────────────────────────────────
    with st.expander("Open Interest  ·  Old vs New Crop", expanded=False):
        dates = old.index.union(other.index).sort_values()
        fig_oi = go.Figure([
            go.Bar(x=dates, y=old.reindex(dates)["Total OI"]/1000, name="Old Crop",
                   marker=dict(color=C_OLD,opacity=0.85,line=dict(width=0)),
                   hovertemplate="<b>%{x|%d %b %y}</b><br>Old OI: %{y:.1f}k<extra></extra>"),
            go.Bar(x=dates, y=other.reindex(dates)["Total OI"]/1000, name="New Crop",
                   marker=dict(color=C_NEW,opacity=0.85,line=dict(width=0)),
                   hovertemplate="<b>%{x|%d %b %y}</b><br>New OI: %{y:.1f}k<extra></extra>"),
        ])
        fig_oi.update_layout(**_BASE, barmode="stack", height=300,
            title=dict(text="Open Interest — Old vs New Crop  ·  k lots",font=dict(size=12,color="#444"),x=0),
            margin=dict(l=50,r=12,t=38,b=68), bargap=0.18,
            legend=dict(orientation="h",y=-0.24,x=0.5,xanchor="center",font_size=10),
            xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
            yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
        st.plotly_chart(fig_oi, width='stretch')

    # ── Net positions ─────────────────────────────────────────────────────────
    with st.expander("Net Positions  ·  MM & Commercial", expanded=False):
        c1, c2 = st.columns(2)
        for col_c, (col, title) in zip([c1,c2],[("MM Net","Managed Money Net"),("Comm Net","Commercial (Prod) Net")]):
            with col_c:
                fig = go.Figure()
                for crop_df, lbl, clr in [(old,"Old Crop",C_OLD),(other,"New Crop",C_NEW)]:
                    if col in crop_df.columns:
                        fig.add_trace(go.Scatter(x=crop_df.index, y=crop_df[col]/1000, name=lbl,
                            line=dict(color=clr, width=2.2, shape="spline", smoothing=0.6),
                            hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>{lbl}: %{{y:.1f}}k<extra></extra>"))
                fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.15)")
                fig.update_layout(**_BASE, height=340,
                    title=dict(text=f"{title}  ·  k lots",font=dict(size=12,color="#444"),x=0),
                    margin=dict(l=50,r=20,t=40,b=70),
                    legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",font_size=10,bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                    yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
                st.plotly_chart(fig, width='stretch')

    # ── Gross legs ────────────────────────────────────────────────────────────
    with st.expander("Gross Legs  ·  Old vs New Crop", expanded=False):
        gross_legs = [(c,t) for c,t in [
            ("MM Long","MM Long"),("MM Short","MM Short"),
            ("Producer Long","Comm Long"),("Producer Short","Comm Short"),
        ] if c in old.columns or c in other.columns]
        for col, title in gross_legs:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                for crop_df, lbl, clr in [(old,"Old",C_OLD),(other,"New",C_NEW)]:
                    if col in crop_df.columns:
                        fig.add_trace(go.Scatter(x=crop_df.index, y=crop_df[col]/1000, name=lbl,
                            line=dict(color=clr,width=2.2,shape="spline",smoothing=0.6),
                            hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>{lbl}: %{{y:.1f}}k<extra></extra>"))
                fig.update_layout(**_BASE, height=300,
                    title=dict(text=f"{title}  ·  k lots",font=dict(size=12,color="#444"),x=0),
                    margin=dict(l=50,r=20,t=40,b=70),
                    legend=dict(orientation="h",y=-0.24,x=0.5,xanchor="center",font_size=10,bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                    yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
                st.plotly_chart(fig, width='stretch')
            with c2:
                dates2 = old.index.union(other.index).sort_values()
                fig2 = go.Figure([
                    go.Bar(x=dates2, y=old.reindex(dates2)[col]/1000 if col in old.columns else None,
                           name="Old", marker=dict(color=C_OLD,opacity=0.85,line=dict(width=0)),
                           hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>Old: %{{y:.1f}}k<extra></extra>"),
                    go.Bar(x=dates2, y=other.reindex(dates2)[col]/1000 if col in other.columns else None,
                           name="New", marker=dict(color=C_NEW,opacity=0.85,line=dict(width=0)),
                           hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>New: %{{y:.1f}}k<extra></extra>"),
                ])
                fig2.update_layout(**_BASE, barmode="stack", height=280,
                    title=dict(text=f"{title} — Stacked  ·  k lots",font=dict(size=12,color="#444"),x=0),
                    margin=dict(l=50,r=20,t=40,b=70), bargap=0.12,
                    legend=dict(orientation="h",y=-0.26,x=0.5,xanchor="center",font_size=10,bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
                    yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
                st.plotly_chart(fig2, width='stretch')




# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRADERS
# ══════════════════════════════════════════════════════════════════════════════
CIT_TRADER_GROUPS = {
    "Spec":          ["Traders Spec Long","Traders Spec Short","Traders Spec Spread"],
    "Commercial":    ["Traders Comm Long","Traders Comm Short"],
    "Index":         ["Traders Index Long","Traders Index Short"],
    "All Reportable":["Traders Tot Rept Long","Traders Tot Rept Short"],
}
DISAGG_TRADER_GROUPS = {
    "Managed Money": ["Traders MM Long","Traders MM Short","Traders MM Spread"],
    "Swap Dealers":  ["Traders Swap Long","Traders Swap Short","Traders Swap Spread"],
    "Other Rept":    ["Traders Other Long","Traders Other Short","Traders Other Spread"],
    "Producer":      ["Traders Producer Long","Traders Producer Short"],
    "All Reportable":["Traders Tot Rept Long","Traders Tot Rept Short","Traders Total"],
}
TRADER_COLORS = [C_LONG, C_SHORT, "#94a3b8", C_NET, C_PRICE]

def render_traders(d, report, color):
    grp_map = CIT_TRADER_GROUPS if report=="CIT" else DISAGG_TRADER_GROUPS
    all_t   = [c for g in grp_map.values() for c in g if c in d.columns]

    latest = d.iloc[-1]
    kpi_items = [(c.replace("Traders ",""), f"{int(latest.get(c,0))}", "")
                 for c in all_t[:8] if pd.notna(latest.get(c))]
    kpi_row(kpi_items, color)

    group = st.selectbox("Group", list(grp_map.keys()), key="traders_grp")
    sel_cols = [c for c in grp_map[group] if c in d.columns]
    nice     = [c.replace("Traders ","") for c in sel_cols]

    fig = go.Figure()
    for col, name, clr in zip(sel_cols, nice, TRADER_COLORS):
        fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=name,
            line=dict(color=clr,width=2.0),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{name}: %{{y:.0f}}<extra></extra>"))
    fig.update_layout(
        **_BASE, height=360,
        title=dict(text=f"Traders in Each Category — {group}",font=dict(size=12,color="#333"),x=0),
        margin=dict(l=50,r=20,t=42,b=70),
        legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",font_size=10),
        xaxis=dict(**_ax(x=True),tickformat="%b '%y"),
        yaxis=dict(**_ax(),title_text="# traders",title_font_size=10))
    st.plotly_chart(fig, width='stretch')

    with st.expander("Weekly change — trader counts", expanded=False):
        cols_w = st.columns(min(len(sel_cols),3))
        for i,(col,name) in enumerate(zip(sel_cols,nice)):
            with cols_w[i%3]:
                chg = d[col].diff().tail(13)
                dates_b = d["Date"].tail(13)
                fb = go.Figure(go.Bar(x=dates_b, y=chg,
                    marker=dict(color=[C_LONG if v>=0 else C_SHORT for v in chg],
                                opacity=0.82,line=dict(width=0)),
                    hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>Δ: %{{y:+.0f}}<extra></extra>"))
                fb.add_hline(y=0,line_width=1,line_color="rgba(0,0,0,0.14)")
                fb.update_layout(**_BASE,height=240,
                    title=dict(text=f"{name} — Δ",font=dict(size=10,color="#444"),x=0),
                    margin=dict(l=40,r=8,t=32,b=60),showlegend=False,
                    xaxis=dict(**_ax(x=True),tickformat="%d %b"),
                    yaxis=dict(**_ax()))
                st.plotly_chart(fb, width='stretch')

    show_table(d, all_t, sel_cols, "Data table — trader counts", scale=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CONCENTRATION
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — RECAP
# ══════════════════════════════════════════════════════════════════════════════
_RECAP_GROUP_BG = {
    "Gross Positions": "#d1d5db",
    "NET":             "#bae6fd",
    "SPREAD":          "#fed7aa",
    "SP":              "#fed7aa",
    "Spec ex Swap":    "#a7f3d0",
    "OI":              "#e5e7eb",
    "OI · k lots":     "#e5e7eb",
    "Δ 1w":            "#f9a8d4",
    "OI %":            "#1e3a8a",
    "Nominal (M USD)": "#d1fae5",
    "Nominal (M GBP)": "#fef9c3",
    "# Traders":       "#ede9fe",
    "k lots / Trader": "#fef08a",
    "Old Crop":        "#fde68a",
    "New Crop":        "#bbf7d0",
    "Net · k lots":    "#bae6fd",
    "MM Long":         "#d1fae5",
    "MM Short":        "#fee2e2",
    "Prod Long":       "#e0e7ff",
    "Prod Short":      "#fce7f3",
    "Rollex Px":       "#fef3c7",
}
_RECAP_GROUP_TEXT = {
    "OI %": "#ffffff",
}
_CHANGE_BG = "#f9a8d4"

_RECAP_CSS = """
<style>
.rtbl{border-collapse:collapse;font-size:.76rem;width:100%;font-family:-apple-system,sans-serif}
.rtbl th,.rtbl td{border:1px solid #e5e7eb;padding:3px 8px;white-space:nowrap;text-align:center}
.rtbl .grp{text-align:center;font-weight:700;font-size:.75rem;letter-spacing:.04em}
.rtbl .idx{text-align:left;font-weight:600;color:#374151;background:#f9fafb;min-width:70px}
.rtbl .sub{background:#f9fafb;font-size:.70rem;color:#555;font-weight:600;text-align:center}
.rtbl tbody tr:hover td{background:#f0f9ff!important}
.rpos{color:#16a34a}.rneg{color:#dc2626}
</style>
"""

def _recap_html(df, signed=False, change_table=False, scroll=False, signed_groups=None, pct_groups=None, pct_subcols=None):
    if df.empty: return ""
    cols = list(df.columns)
    # Build group spans
    groups, prev = [], None
    for c in cols:
        g = c[0]
        if g == prev: groups[-1][1] += 1
        else: groups.append([g, 1]); prev = g

    # Header row 1 — merged group headers
    h1 = '<tr><th class="idx sub"></th>'
    for g, span in groups:
        bg = _RECAP_GROUP_BG.get(g, "#f9fafb")
        fg = _RECAP_GROUP_TEXT.get(g, "#111827")
        h1 += f'<th colspan="{span}" class="grp" style="background:{bg};color:{fg}">{g}</th>'
    h1 += '</tr>'

    # Header row 2 — sub-column names
    h2 = '<tr><th class="idx sub"></th>'
    for c in cols:
        g = c[0]
        if g in _RECAP_GROUP_TEXT:
            bg = _RECAP_GROUP_BG.get(g, "#f9fafb")
            fg = _RECAP_GROUP_TEXT[g]
            h2 += f'<th class="sub" style="background:{bg};color:{fg}">{c[1]}</th>'
        else:
            h2 += f'<th class="sub">{c[1]}</th>'
    h2 += '</tr>'

    # Body rows
    body = ""
    for idx, row in df.iterrows():
        body += f'<tr><td class="idx">{idx}</td>'
        for c in cols:
            v = row[c]
            if pd.isna(v): body += '<td>—</td>'; continue
            use_signed = signed or change_table or (
                signed_groups and isinstance(c, tuple) and c[0] in signed_groups)
            use_pct = ((pct_groups and isinstance(c, tuple) and c[0] in pct_groups) or
                       (pct_subcols and isinstance(c, tuple) and c in pct_subcols))
            if use_signed:
                txt = f"{v:+.1f}"
                cls = "rpos" if v > 0 else ("rneg" if v < 0 else "")
            elif use_pct:
                txt = f"{v:.1f}%"; cls = ""
            else:
                txt = f"{v:.1f}"; cls = ""
            body += f'<td class="{cls}">{txt}</td>'
        body += '</tr>'

    scroll_style = "overflow-x:auto;overflow-y:auto;max-height:420px;" if scroll else "overflow-x:auto;"
    return (f'{_RECAP_CSS}<div style="{scroll_style}margin-bottom:6px">'
            f'<table class="rtbl"><thead>{h1}{h2}</thead>'
            f'<tbody>{body}</tbody></table></div>')

def _build_recap_df(d, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)

    cols = {}

    if report == "CIT":
        for src, dst in [
            ("Spec Long",    "Large Long"),  ("Spec Short",    "Large Short"),
            ("Non Rep Long", "Small Long"),  ("Non Rep Short", "Small Short"),
            ("Index Long",   "Index Long"),  ("Index Short",   "Index Short"),
            ("Comm Long",    "Comm Long"),   ("Comm Short",    "Comm Short"),
        ]:
            if src in d.columns:
                cols[("Gross Positions", dst)] = gc(src) / 1000

        cols[("NET", "Large")]        = gc("Spec Net")   / 1000
        cols[("NET", "Small")]        = gc("Non Rep Net") / 1000
        cols[("NET", "Index")]        = gc("Index Net")   / 1000
        cols[("NET", "Comm")]         = gc("Comm Net")    / 1000
        cols[("NET", "Large+Small")]  = (gc("Spec Net") + gc("Non Rep Net")) / 1000
        cols[("NET", "Lrg+Sml+Idx")] = (gc("Spec Net") + gc("Non Rep Net") + gc("Index Net")) / 1000

        if "Spec Spread" in d.columns:
            cols[("SPREAD", "Spec Spread")] = gc("Spec Spread") / 1000

        cols[("OI", "Total OI")] = gc("Total OI") / 1000

    else:  # Disagg
        for src, dst in [
            ("MM Long",       "MM Long"),    ("MM Short",       "MM Short"),
            ("Other Long",    "Other Long"), ("Other Short",    "Other Short"),
            ("Non Rep Long",  "Non-Rep Long"),("Non Rep Short", "Non-Rep Short"),
            ("Swap Long",     "Swap Long"),  ("Swap Short",     "Swap Short"),
            ("Producer Long", "Comm Long"),  ("Producer Short", "Comm Short"),
        ]:
            if src in d.columns:
                cols[("Gross Positions", dst)] = gc(src) / 1000

        cols[("Spec ex Swap", "Long")]  = (gc("MM Long")  + gc("Other Long")  + gc("Non Rep Long"))  / 1000
        cols[("Spec ex Swap", "Short")] = (gc("MM Short") + gc("Other Short") + gc("Non Rep Short")) / 1000

        cols[("NET", "MM")]   = gc("MM Net")   / 1000
        cols[("NET", "Rest")] = (gc("Other Net") + gc("Non Rep Net")) / 1000
        cols[("NET", "Swap")] = gc("Swap Net")  / 1000
        cols[("NET", "Comm")] = gc("Comm Net")  / 1000

        for src, dst in [
            ("MM Spread",    "MM Spread"),
            ("Other Spread", "Other Spread"),
            ("Swap Spread",  "Swap Spread"),
        ]:
            if src in d.columns:
                cols[("SP", dst)] = gc(src) / 1000

        cols[("OI", "Total OI")] = gc("Total OI") / 1000

    # Add Rollex Px level — diff in summary gives absolute price change
    cols[("Rollex Px", "Level")] = gc("Px")

    body = pd.DataFrame(cols)
    body.index = pd.to_datetime(d["Date"])
    body = body.iloc[::-1]  # newest first

    row_1w, row_4w = {}, {}
    for c in body.columns:
        if len(body) >= 2:
            row_1w[c] = body.iloc[0][c] - body.iloc[1][c]
        if len(body) >= 5:
            row_4w[c] = body.iloc[0][c] - body.iloc[4][c]

    summary = pd.DataFrame([row_1w, row_4w],
                           index=["+/-1w", "+/-4w"],
                           columns=body.columns)

    # Rollex Px Δ% 1w — body (newest first): pct_change(-1) = row vs the one after (older)
    px_lvl = body[("Rollex Px", "Level")]
    body[("Rollex Px", "Δ% 1w")] = px_lvl.pct_change(-1) * 100

    # Override summary Δ% 1w with proper cumulative % change (not diff of pct)
    summary[("Rollex Px", "Δ% 1w")] = np.nan
    if len(px_lvl) >= 2 and px_lvl.iloc[1] != 0:
        summary.loc["+/-1w", ("Rollex Px", "Δ% 1w")] = (px_lvl.iloc[0] / px_lvl.iloc[1] - 1) * 100
    if len(px_lvl) >= 5 and px_lvl.iloc[4] != 0:
        summary.loc["+/-4w", ("Rollex Px", "Δ% 1w")] = (px_lvl.iloc[0] / px_lvl.iloc[4] - 1) * 100

    body.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in body.index]
    return summary, body


def _build_oi_df(d, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()

    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)

    total_oi = gc("Total OI") / 1000

    if report == "CIT":
        oi_cols = {
            "Large Spec": (gc("Spec Long") + gc("Spec Short")) / 2 / 1000 + gc("Spec Spread") / 1000,
            "Small Spec": (gc("Non Rep Long") + gc("Non Rep Short")) / 2 / 1000,
            "Index":      (gc("Index Long") + gc("Index Short")) / 2 / 1000,
            "Commercial": (gc("Comm Long") + gc("Comm Short")) / 2 / 1000,
        }
    else:
        oi_cols = {
            "MM":         (gc("MM Long") + gc("MM Short")) / 2 / 1000 + gc("MM Spread") / 1000,
            "Other":      (gc("Other Long") + gc("Other Short")) / 2 / 1000 + gc("Other Spread") / 1000,
            "Swap":       (gc("Swap Long") + gc("Swap Short")) / 2 / 1000 + gc("Swap Spread") / 1000,
            "Commercial": (gc("Producer Long") + gc("Producer Short")) / 2 / 1000,
            "Non-Rep":    (gc("Non Rep Long") + gc("Non Rep Short")) / 2 / 1000,
        }

    oi_df = pd.DataFrame(oi_cols)
    oi_df.index = pd.to_datetime(d["Date"])
    oi_df = oi_df.iloc[::-1].iloc[:20]  # newest first, last 20

    total_s = pd.Series(total_oi.values, index=pd.to_datetime(d["Date"])).iloc[::-1].iloc[:20]
    pct_df  = oi_df.div(total_s.values, axis=0) * 100   # % before adding Total OI col

    oi_df["Total OI"] = total_s.values                  # add Total OI after pct calc
    chg_df = oi_df.diff(-1)

    cats     = list(oi_cols.keys())
    all_cats = cats + ["Total OI"]
    combined = pd.concat([oi_df[all_cats], pct_df[cats], chg_df[all_cats]], axis=1)
    combined.columns = pd.MultiIndex.from_tuples(
        [("OI · k lots", c) for c in all_cats] +
        [("OI %",        c) for c in cats] +
        [("Δ 1w",        c) for c in all_cats]
    )
    combined.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in combined.index]
    return combined


def _build_nominal_df(d, commodity, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)

    size = CONTRACT_SIZE.get(commodity, 1)
    unit = CONTRACT_UNIT.get(commodity, "MT")
    ccy  = "GBP" if commodity == "LCC" else "USD"
    px   = gc("Px")

    # mult: M currency per 1 contract
    if unit == "lbs":
        mult = px * size / 100 / 1_000_000   # cents/lb → USD
    else:
        mult = px * size / 1_000_000          # USD or GBP per MT

    grp = f"Nominal (M {ccy})"

    if report == "CIT":
        cols = {
            (grp, "Spec Net"):   (gc("Spec Net") + gc("Non Rep Net").fillna(0)) * mult,
            (grp, "Net Index"):  gc("Index Net") * mult,
            (grp, "Comm Long"):  gc("Comm Long") * mult,
            (grp, "Comm Short"): gc("Comm Short") * mult,
        }
    else:
        cols = {
            (grp, "Spec Net"):    (gc("MM Net") + gc("Other Net").fillna(0) + gc("Non Rep Net").fillna(0)) * mult,
            (grp, "Swap Net"):    gc("Swap Net")       * mult,
            (grp, "Comm Long"):   gc("Producer Long")  * mult,
            (grp, "Comm Short"):  gc("Producer Short") * mult,
        }

    body = pd.DataFrame(cols)
    body.index = pd.to_datetime(d["Date"])
    body = body.iloc[::-1].iloc[:20]

    row_1w, row_4w = {}, {}
    for c in body.columns:
        if len(body) >= 2: row_1w[c] = body.iloc[0][c] - body.iloc[1][c]
        if len(body) >= 5: row_4w[c] = body.iloc[0][c] - body.iloc[4][c]

    summary = pd.DataFrame([row_1w, row_4w], index=["+/-1w", "+/-4w"], columns=body.columns)
    body.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in body.index]
    return summary, body


def _summary_and_body(cols_dict, d_index, n=20):
    body = pd.DataFrame(cols_dict)
    body.index = pd.to_datetime(d_index)
    body = body.iloc[::-1].iloc[:n]
    row_1w, row_4w = {}, {}
    for c in body.columns:
        if len(body) >= 2: row_1w[c] = body.iloc[0][c] - body.iloc[1][c]
        if len(body) >= 5: row_4w[c] = body.iloc[0][c] - body.iloc[4][c]
    summary = pd.DataFrame([row_1w, row_4w], index=["+/-1w", "+/-4w"], columns=body.columns)
    body.index = [f"{dt.day}-{dt.strftime('%b-%y')}" for dt in body.index]
    return summary, body


def _build_traders_df(d, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty: return pd.DataFrame(), pd.DataFrame()
    def gc(name): return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)

    if report == "CIT":
        cols = {
            ("# Traders", "Lrg Long"):  gc("Traders Spec Long"),
            ("# Traders", "Lrg Short"): gc("Traders Spec Short"),
            ("# Traders", "Idx Long"):  gc("Traders Index Long"),
            ("# Traders", "Idx Short"): gc("Traders Index Short"),
            ("# Traders", "Lrg Spread"):gc("Traders Spec Spread"),
            ("# Traders", "Comm Long"): gc("Traders Comm Long"),
            ("# Traders", "Comm Short"):gc("Traders Comm Short"),
        }
    else:
        cols = {
            ("# Traders", "MM Long"):    gc("Traders MM Long"),
            ("# Traders", "MM Short"):   gc("Traders MM Short"),
            ("# Traders", "MM Spread"):  gc("Traders MM Spread"),
            ("# Traders", "Swap Long"):  gc("Traders Swap Long"),
            ("# Traders", "Swap Short"): gc("Traders Swap Short"),
            ("# Traders", "Swap Spread"):gc("Traders Swap Spread"),
            ("# Traders", "Other Long"): gc("Traders Other Long"),
            ("# Traders", "Other Short"):gc("Traders Other Short"),
            ("# Traders", "Prod Long"):  gc("Traders Producer Long"),
            ("# Traders", "Prod Short"): gc("Traders Producer Short"),
        }
    return _summary_and_body(cols, d["Date"])


def _build_lots_per_trader_df(d, report):
    d = d.sort_values("Date", ascending=True).reset_index(drop=True)
    if d.empty: return pd.DataFrame(), pd.DataFrame()
    def gc(name): return d[name].astype(float) if name in d.columns else pd.Series(0.0, index=d.index)
    def safe_div(num, den): return num.where(den == 0, num / den.replace(0, np.nan))

    if report == "CIT":
        cols = {
            ("k lots / Trader", "Lrg Long"):  safe_div(gc("Spec Long")  / 1000, gc("Traders Spec Long")),
            ("k lots / Trader", "Lrg Short"): safe_div(gc("Spec Short") / 1000, gc("Traders Spec Short")),
            ("k lots / Trader", "Idx Long"):  safe_div(gc("Index Long") / 1000, gc("Traders Index Long")),
            ("k lots / Trader", "Idx Short"): safe_div(gc("Index Short")/ 1000, gc("Traders Index Short")),
            ("k lots / Trader", "Lrg Spread"):safe_div(gc("Spec Spread")/ 1000, gc("Traders Spec Spread")),
            ("k lots / Trader", "Comm Long"): safe_div(gc("Comm Long")  / 1000, gc("Traders Comm Long")),
            ("k lots / Trader", "Comm Short"):safe_div(gc("Comm Short") / 1000, gc("Traders Comm Short")),
        }
    else:
        cols = {
            ("k lots / Trader", "MM Long"):    safe_div(gc("MM Long")       / 1000, gc("Traders MM Long")),
            ("k lots / Trader", "MM Short"):   safe_div(gc("MM Short")      / 1000, gc("Traders MM Short")),
            ("k lots / Trader", "MM Spread"):  safe_div(gc("MM Spread")     / 1000, gc("Traders MM Spread")),
            ("k lots / Trader", "Swap Long"):  safe_div(gc("Swap Long")     / 1000, gc("Traders Swap Long")),
            ("k lots / Trader", "Swap Short"): safe_div(gc("Swap Short")    / 1000, gc("Traders Swap Short")),
            ("k lots / Trader", "Swap Spread"):safe_div(gc("Swap Spread")   / 1000, gc("Traders Swap Spread")),
            ("k lots / Trader", "Other Long"): safe_div(gc("Other Long")    / 1000, gc("Traders Other Long")),
            ("k lots / Trader", "Other Short"):safe_div(gc("Other Short")   / 1000, gc("Traders Other Short")),
            ("k lots / Trader", "Prod Long"):  safe_div(gc("Producer Long") / 1000, gc("Traders Producer Long")),
            ("k lots / Trader", "Prod Short"): safe_div(gc("Producer Short")/ 1000, gc("Traders Producer Short")),
        }
    return _summary_and_body(cols, d["Date"])


def render_recap(d, report, color, commodity="KC", is_options=False):
    if d.empty:
        st.warning("No data for the selected filters."); return

    summary, body = _build_recap_df(d, report)
    if body.empty:
        st.warning("No data."); return

    view = body.iloc[:20]

    _PX_PCT = {("Rollex Px", "Δ% 1w")}

    with st.expander("Change summary  ·  k lots", expanded=True):
        st.markdown(_recap_html(summary, signed=True, pct_subcols=_PX_PCT), unsafe_allow_html=True)

    with st.expander("Historical positions  ·  k lots", expanded=True):
        st.markdown(_recap_html(view, scroll=True, pct_subcols=_PX_PCT), unsafe_allow_html=True)

    with st.expander("Weekly change  ·  k lots", expanded=True):
        chg = view.diff(-1)
        st.markdown(_recap_html(chg, signed=True, change_table=True, scroll=True, pct_subcols=_PX_PCT), unsafe_allow_html=True)

    oi_tbl = _build_oi_df(d, report)
    with st.expander("OI by category  ·  k lots  &  %", expanded=False):
        st.markdown(_recap_html(oi_tbl, signed_groups={"Δ 1w"}, pct_groups={"OI %"}, scroll=True), unsafe_allow_html=True)

    nom_summary, nom_body = _build_nominal_df(d, commodity, report)
    if not nom_body.empty:
        ccy = "GBP" if commodity == "LCC" else "USD"
        with st.expander(f"Nominal Exposure  ·  M {ccy}", expanded=False):
            _spec_note = ("Spec Net = MM Net + Other Net + Non-Rep Net"
                          if report != "CIT" else
                          "Spec Net = Large Spec Net + Non-Rep Net")
            st.markdown(
                f"<div style='font-size:.72rem;color:#6b7280;margin-bottom:6px'>"
                f"{_spec_note}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(_recap_html(nom_summary, signed=True), unsafe_allow_html=True)
            st.markdown(_recap_html(nom_body, scroll=True), unsafe_allow_html=True)

    tr_summary, tr_body = _build_traders_df(d, report)
    if not tr_body.empty:
        with st.expander("# of Traders", expanded=False):
            st.markdown(_recap_html(tr_summary, signed=True), unsafe_allow_html=True)
            st.markdown(_recap_html(tr_body, scroll=True), unsafe_allow_html=True)

    lpt_summary, lpt_body = _build_lots_per_trader_df(d, report)
    if not lpt_body.empty:
        with st.expander("k lots / Trader  (avg position size per trader)", expanded=False):
            st.markdown(_recap_html(lpt_summary, signed=True), unsafe_allow_html=True)
            st.markdown(_recap_html(lpt_body, scroll=True), unsafe_allow_html=True)

    if report == "CIT":
        guide = """
**Large Long / Large Short** — Non-Commercial

**Small Long / Small Short** — Non-Reportable

**Large+Small** — Non-Commercial Net + Non-Reportable Net (total non-index speculative net)

**Lrg+Sml+Idx** — Non-Commercial Net + Non-Reportable Net + Index Traders Net (everything ex-Commercial)
"""
    else:
        guide = """
**Spec ex Swap Long/Short** — MM Long/Short + Other Long/Short + Non-Rep Long/Short (all speculative ex swap dealers)

**Rest (NET)** — Other Net + Non-Reportable Net combined
"""
    with st.expander("Column guide", expanded=False):
        st.markdown(guide)


# ── CIT vs Disagg crosswalk ───────────────────────────────────────────────────
CROSSWALK = {
    "Non-Comm vs Managed Money": {
        "cit_long":"Spec Long",    "cit_short":"Spec Short",    "cit_net":"Spec Net",
        "dag_long":"MM Long",      "dag_short":"MM Short",      "dag_net":"MM Net",
        "cit_label":"Non-Commercial (CIT)", "dag_label":"Managed Money (Disagg)",
        "desc":"",
    },
    "Index Traders vs Swap Dealers": {
        "cit_long":"Index Long",   "cit_short":"Index Short",   "cit_net":"Index Net",
        "dag_long":"Swap Long",    "dag_short":"Swap Short",    "dag_net":"Swap Net",
        "cit_label":"Index Traders (CIT)", "dag_label":"Swap Dealers (Disagg)",
        "desc":"",
    },
    "Commercial vs Producer/Merchant": {
        "cit_long":"Comm Long",     "cit_short":"Comm Short",     "cit_net":"Comm Net",
        "dag_long":"Producer Long", "dag_short":"Producer Short", "dag_net":"Comm Net",
        "cit_label":"CIT Commercial", "dag_label":"Producer/Merchant (Disagg)",
        "desc":"Physical hedgers — CIT Commercial includes swap-dealer activity that Disagg separates out.",
    },
    "CIT (NonComm + Index) vs Disagg (MM + Swap + Others)": {
        "cit_long":None, "cit_short":None, "cit_net":"Fin Net",
        "dag_long":None, "dag_short":None, "dag_net":"Fin Net",
        "cit_label":"CIT NonComm + Index", "dag_label":"Disagg MM + Swap + Other",
        "desc":"Total non-physical/financial side — CIT (NonComm+Index) vs Disagg (MM+Swap+Other).",
        "computed":True,
    },
}

CONC_COLS = ["Conc Gross 4 Long","Conc Gross 4 Short",
             "Conc Gross 8 Long","Conc Gross 8 Short",
             "Conc Net 4 Long","Conc Net 4 Short",
             "Conc Net 8 Long","Conc Net 8 Short"]

def render_concentration(d, color):
    avail = [c for c in CONC_COLS if c in d.columns]
    if not avail: st.info("No concentration data available."); return

    latest = d.iloc[-1]

    # Summary table
    st.markdown(f"**Latest — week of {pd.to_datetime(latest['Date']).strftime('%d %b %Y')}**")
    rows = {
        "":         ["4 Traders","8 Traders"],
        "Gross Long": [f"{latest.get('Conc Gross 4 Long',np.nan):.1f}%",
                       f"{latest.get('Conc Gross 8 Long',np.nan):.1f}%"],
        "Gross Short":[f"{latest.get('Conc Gross 4 Short',np.nan):.1f}%",
                       f"{latest.get('Conc Gross 8 Short',np.nan):.1f}%"],
        "Net Long":  [f"{latest.get('Conc Net 4 Long',np.nan):.1f}%",
                      f"{latest.get('Conc Net 8 Long',np.nan):.1f}%"],
        "Net Short": [f"{latest.get('Conc Net 4 Short',np.nan):.1f}%",
                      f"{latest.get('Conc Net 8 Short',np.nan):.1f}%"],
    }
    st.dataframe(pd.DataFrame(rows).set_index(""), width='content', height=130)
    st.markdown("")

    sel = st.multiselect("Series to chart", avail,
                         default=["Conc Gross 4 Long","Conc Gross 4 Short",
                                  "Conc Gross 8 Long","Conc Gross 8 Short"],
                         key="conc_sel")
    CONC_PALETTE = ["#1a56db","#dc2626","#1a56db","#dc2626",
                    "#7c3aed","#059669","#7c3aed","#059669"]
    CONC_DASH    = ["solid","solid","dash","dash","solid","solid","dash","dash"]

    if sel:
        fig = go.Figure()
        for i,col in enumerate(sel):
            fig.add_trace(go.Scatter(x=d["Date"], y=d[col],
                name=col.replace("Conc ",""),
                line=dict(color=CONC_PALETTE[i%8], width=2.0, dash=CONC_DASH[i%8]),
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{col}: %{{y:.1f}}%<extra></extra>"))
        fig.update_layout(
            **_BASE, height=360,
            title=dict(text="Concentration — % of OI held by largest traders",
                       font=dict(size=12,color="#333"),x=0),
            margin=dict(l=50,r=20,t=42,b=70),
            legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",font_size=10),
            xaxis=dict(**_ax(x=True),tickformat="%b '%y"),
            yaxis=dict(**_ax(),title_text="% of OI",title_font_size=10))
        st.plotly_chart(fig, width='stretch')

    show_table(d, avail, avail[:4], "Data table — Concentration ratios", scale=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — EXPOSURE (Nominal; VaR placeholder)
# ══════════════════════════════════════════════════════════════════════════════
def render_exposure(d, commodity, color):
    cs  = CONTRACT_SIZE[commodity]
    cu  = CONTRACT_UNIT[commodity]
    px  = d["Px"].dropna().iloc[-1] if "Px" in d.columns and not d["Px"].dropna().empty else None

    st.markdown(
        f"<div style='background:#f0f4ff;border:1px solid #c7d7ff;border-radius:8px;"
        f"padding:12px 18px;margin-bottom:14px;font-size:.84rem'>"
        f"<b>Contract spec</b>: {commodity} — {cs:,} {cu} per contract &nbsp;|&nbsp; "
        f"Latest price: {'%.2f' % px if px else '—'}"
        f"</div>", unsafe_allow_html=True)

    if px is None:
        st.warning("Price data unavailable — cannot compute nominal.")
        return

    net_map = {}
    for c in ["Spec Net","MM Net","Combined Spec Net","Comm Net","Non Rep Net","Total OI"]:
        if c in d.columns:
            net_map[c] = d[c].iloc[-1]

    rows = []
    for col, lots in net_map.items():
        if pd.isna(lots): continue
        nominal = abs(lots) * cs * px
        rows.append({"Position": col, "Contracts": f"{lots/1000:.1f}k",
                     "Nominal (USD)": f"${nominal/1e6:.1f}M"})
    if rows:
        st.markdown("**Nominal Exposure — latest week**")
        st.dataframe(pd.DataFrame(rows).set_index("Position"), width='content')

    # Timeseries of nominal net spec
    nc = next((c for c in ["Spec Net","MM Net","Combined Spec Net"] if c in d.columns), None)
    if nc:
        nom_ts = d[nc].abs() * cs * d["Px"].ffill() / 1e6
        fig = go.Figure(go.Scatter(
            x=d["Date"], y=nom_ts, name="Nominal |Net|",
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.09)",
            line=dict(color=color,width=2.0),
            hovertemplate="<b>%{x|%b %Y}</b><br>Nominal: $%{y:.0f}M<extra></extra>"))
        fig.update_layout(
            **_BASE, height=320,
            title=dict(text=f"{nc} — Nominal Exposure  ·  $M",font=dict(size=12,color="#333"),x=0),
            margin=dict(l=60,r=20,t=42,b=50), showlegend=False,
            xaxis=dict(**_ax(x=True),tickformat="%b '%y"),
            yaxis=dict(**_ax(),title_text="$M",title_font_size=10))
        st.plotly_chart(fig, width='stretch')

    st.markdown(
        "<div style='background:#fff8e8;border:1px solid #fde68a;border-radius:8px;"
        "padding:10px 16px;margin-top:10px;font-size:.82rem;color:#92400e'>"
        "VaR module — reserved for integration from the existing VaR project.</div>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def render_analysis(d, report, color):
    if report == "CIT":
        _net  = ["Spec Net","Comm Net","Index Net","Non Rep Net","Combined Spec Net"]
        _gross = ["Spec Long","Spec Short","Non Rep Long","Non Rep Short",
                  "Index Long","Index Short","Comm Long","Comm Short"]
    else:
        _net  = ["MM Net","Comm Net","Swap Net","Other Net","Non Rep Net","Combined Spec Net"]
        _gross = ["MM Long","MM Short","Producer Long","Producer Short",
                  "Swap Long","Swap Short","Other Long","Other Short",
                  "Non Rep Long","Non Rep Short"]
    all_opts = [c for c in _net + _gross if c in d.columns]

    st.markdown("#### Price vs Positioning")
    c1,_ = st.columns([2,5])
    with c1: sel = st.selectbox("COT element", all_opts, key="anal_col")

    if sel and "Px" in d.columns:
        ch1,ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(scatter_2d(d,"Px",sel,color,
                f"Price Δ%  vs  {sel} Δ","Price weekly Δ%",f"{sel} Δ (k lots)"),
                width='stretch')
        with ch2:
            x = np.asarray(d["Px"], dtype=float)
            y = np.asarray(d[sel], dtype=float) / 1000
            dates = np.asarray(d["Date"])
            mask = ~(np.isnan(x)|np.isnan(y))
            if mask.sum()>=5:
                r2 = float(np.corrcoef(x[mask],y[mask])[0,1]**2)
                sl,ic = np.polyfit(x[mask],y[mask],1)
                xl = np.linspace(x[mask].min(),x[mask].max(),200)
                rec = (dates[mask]-dates[mask].min()).astype("timedelta64[D]").astype(float)
                nr  = rec/max(rec.max(),1)
                rv,gv,bv = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x[mask],y=y[mask],mode="markers",
                    marker=dict(color=nr,
                        colorscale=[[0,"rgba(200,210,230,0.5)"],[1,f"rgba({rv},{gv},{bv},0.85)"]],
                        size=7,line=dict(width=0.5,color="white")),
                    text=pd.to_datetime(dates[mask]).strftime("%Y-%m-%d"),
                    hovertemplate=f"<b>%{{text}}</b><br>Rollex Px: %{{x:.2f}}<br>{sel}: %{{y:.1f}}k<extra></extra>",
                    showlegend=False))
                fig2.add_trace(go.Scatter(x=xl,y=sl*xl+ic,mode="lines",
                    line=dict(color=color,width=1.6,dash="dash"),showlegend=False))
                fig2.add_trace(go.Scatter(x=[x[mask][-1]],y=[y[mask][-1]],mode="markers",
                    showlegend=False,
                    marker=dict(symbol="star",size=14,color=C_SHORT,
                                line=dict(width=1.2,color="white"))))
                fig2.update_layout(
                    **_BASE, height=340,
                    title=dict(text=f"Price Level vs {sel}   "
                               f"<span style='font-size:10px;color:#888'>R²={r2:.2f}</span>",
                               font=dict(size=12,color="#333"),x=0),
                    margin=dict(l=52,r=20,t=48,b=48),
                    xaxis=dict(**_ax(x=True),title_text="Rollex Px"),
                    yaxis=dict(**_ax(),title_text=f"{sel} (k lots)"))
                st.plotly_chart(fig2, width='stretch')

    with st.expander("COT vs COT Cross-Scatter", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            xs_sel = st.multiselect("X axis  (summed if multiple)", all_opts,
                                    default=[all_opts[0]], key="xs_x")
        with c2:
            ys_sel = st.multiselect("Y axis  (summed if multiple)", all_opts,
                                    default=[all_opts[min(1, len(all_opts)-1)]], key="xs_y")
        if xs_sel and ys_sel:
            # Build combined series by summing selected columns
            xs_avail = [c for c in xs_sel if c in d.columns]
            ys_avail = [c for c in ys_sel if c in d.columns]
            if xs_avail and ys_avail:
                d_tmp = d.copy()
                d_tmp["_X"] = sum(d_tmp[c] for c in xs_avail)
                d_tmp["_Y"] = sum(d_tmp[c] for c in ys_avail)
                x_lbl = " + ".join(xs_avail)
                y_lbl = " + ".join(ys_avail)
                st.plotly_chart(scatter_2d(d_tmp, "_X", "_Y", color,
                    f"{x_lbl}  vs  {y_lbl}",
                    f"{x_lbl} Δ (k lots)", f"{y_lbl} Δ (k lots)"),
                    width='stretch')

    # ── 3D helpers shared by both expanders ──────────────────────────────────
    px_opt   = ["Rollex Px"]
    all_3d   = px_opt + all_opts   # Rollex Px always first

    def _build_series(col_list, mode):
        """mode='chg': weekly diff/pct  |  mode='lvl': level"""
        avail = [c for c in col_list if c in d.columns or c == "Rollex Px"]
        if not avail: return pd.Series(dtype=float), ""
        parts = []
        for c in avail:
            s = d["Px"] if c == "Rollex Px" else d[c]
            if mode == "chg":
                parts.append(s.pct_change()*100 if c == "Rollex Px" else s.diff()/1000)
            else:
                parts.append(s if c == "Rollex Px" else s/1000)
        combined = sum(parts)
        lbl = " + ".join(avail)
        return combined, lbl

    with st.expander("3D Scatter — Weekly Change", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1: x3c = st.multiselect("X", all_3d, default=[all_3d[0]], key="3dc_x")
        with c2: y3c = st.multiselect("Y", all_3d, default=[all_3d[1]] if len(all_3d)>1 else [all_3d[0]], key="3dc_y")
        with c3: z3c = st.multiselect("Z", all_3d, default=[all_3d[2]] if len(all_3d)>2 else [all_3d[0]], key="3dc_z")
        if x3c and y3c and z3c:
            xs, xl = _build_series(x3c, "chg")
            ys, yl = _build_series(y3c, "chg")
            zs, zl = _build_series(z3c, "chg")
            if not xs.empty:
                st.plotly_chart(scatter_3d(
                    xs, ys, zs, d["Date"], color,
                    f"{xl}  ×  {yl}  ×  {zl}  — Weekly Δ",
                    f"{xl} Δ", f"{yl} Δ", f"{zl} Δ"),
                    width='stretch')

    with st.expander("3D Scatter — Position Levels", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1: x3l = st.multiselect("X", all_3d, default=[all_3d[0]], key="3dl_x")
        with c2: y3l = st.multiselect("Y", all_3d, default=[all_3d[1]] if len(all_3d)>1 else [all_3d[0]], key="3dl_y")
        with c3: z3l = st.multiselect("Z", all_3d, default=[all_3d[2]] if len(all_3d)>2 else [all_3d[0]], key="3dl_z")
        if x3l and y3l and z3l:
            xs, xl = _build_series(x3l, "lvl")
            ys, yl = _build_series(y3l, "lvl")
            zs, zl = _build_series(z3l, "lvl")
            if not xs.empty:
                st.plotly_chart(scatter_3d(
                    xs, ys, zs, d["Date"], color,
                    f"{xl}  ×  {yl}  ×  {zl}  — Levels",
                    xl, yl, zl),
                    width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — CIT vs DISAGG COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def render_comparison(commodity, start_date, end_date, color):
    if commodity not in CIT_COMMS:
        st.info("CIT vs Disagg comparison is only available for KC, CC, SB, and CT.")
        return

    cit_raw = load_cit()
    dag_raw = load_disagg("F&O")

    cit = cit_raw[
        (cit_raw["Commodity"] == commodity) &
        (cit_raw["Date"] >= pd.Timestamp(start_date)) &
        (cit_raw["Date"] <= pd.Timestamp(end_date))
    ].sort_values("Date").reset_index(drop=True).copy()

    dag = dag_raw[
        (dag_raw["Commodity"] == commodity) &
        (dag_raw["Crop"] == "All") &
        (dag_raw["Date"] >= pd.Timestamp(start_date)) &
        (dag_raw["Date"] <= pd.Timestamp(end_date))
    ].sort_values("Date").reset_index(drop=True).copy()

    if cit.empty or dag.empty:
        st.warning("Insufficient data for the selected range."); return

    # Compute Financial Net
    cit["Fin Net"] = (
        cit.get("Spec Net",  pd.Series(0, index=cit.index)).fillna(0) +
        cit.get("Index Net", pd.Series(0, index=cit.index)).fillna(0)
    )
    dag["Fin Net"] = (
        dag.get("MM Net",    pd.Series(0, index=dag.index)).fillna(0) +
        dag.get("Swap Net",  pd.Series(0, index=dag.index)).fillna(0) +
        dag.get("Other Net", pd.Series(0, index=dag.index)).fillna(0)
    )

    # Controls
    pair = st.selectbox("Pair", list(CROSSWALK.keys()), key="cmp_pair")
    unit = "k lots"
    cfg  = CROSSWALK[pair]
    if cfg.get("desc"):
        st.markdown(
            f"<div style='font-size:.77rem;color:#666;background:#f7f8fa;"
            f"border-radius:6px;padding:7px 12px;margin-bottom:12px'>{cfg['desc']}</div>",
            unsafe_allow_html=True)

    cit_nc, dag_nc = cfg["cit_net"], cfg["dag_net"]
    suffix = "k" if unit == "k lots" else "%"
    ylabel = "k lots" if unit == "k lots" else "% of OI"
    DAG_COLOR = "#64748b"
    r, g, b   = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    # Merge on Date for gap + correlation
    merge_cit = cit[["Date", cit_nc]].rename(columns={cit_nc: "cit_net"}) if cit_nc in cit.columns else None
    merge_dag = dag[["Date", dag_nc]].rename(columns={dag_nc: "dag_net"}) if dag_nc in dag.columns else None
    merged = None
    if merge_cit is not None and merge_dag is not None:
        merged = pd.merge(merge_cit, merge_dag, on="Date", how="inner").sort_values("Date")
        merged["gap"] = (merged["cit_net"] - merged["dag_net"]) / 1000

    # KPI row
    def _net_kpi_cmp(df, col, label):
        if col not in df.columns or df.empty: return (label, "—", "")
        v, p = df[col].iloc[-1], (df[col].iloc[-2] if len(df)>1 else df[col].iloc[-1])
        val  = f"{v/1000:.1f}k" if unit=="k lots" else (
               f"{v/df['Total OI'].iloc[-1]*100:.1f}%" if "Total OI" in df.columns else f"{v/1000:.1f}k")
        chg  = f"{'▲' if v>p else '▼'}{abs(v-p)/1000:.1f}k"
        return (label, val, chg)

    kpi_items = [
        _net_kpi_cmp(cit, cit_nc, cfg["cit_label"]),
        _net_kpi_cmp(dag, dag_nc, cfg["dag_label"]),
    ]
    if merged is not None and not merged.empty:
        gap_latest = merged["gap"].iloc[-1]
        ca  = np.asarray(merged["cit_net"], dtype=float)
        da  = np.asarray(merged["dag_net"], dtype=float)
        msk = ~(np.isnan(ca)|np.isnan(da))
        corr = float(np.corrcoef(ca[msk], da[msk])[0,1]) if msk.sum()>4 else np.nan
        kpi_items += [
            ("Gap (CIT−Disagg)", f"{gap_latest:+.1f}k", ""),
            ("Corr (full period)", f"{corr:.2f}" if pd.notna(corr) else "—", ""),
        ]
    kpi_row(kpi_items, color)

    # ── 1. Overlay Net timeseries ──────────────────────────────────────────────
    traces = []
    if cit_nc in cit.columns:
        traces.append({"trace": go.Scatter(
            x=cit["Date"], y=_get_y(cit, cit_nc, unit), name=cfg["cit_label"],
            line=dict(color=color, width=2.2),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{cfg['cit_label']}: %{{y:.1f}}{suffix}<extra></extra>")})
    if dag_nc in dag.columns:
        traces.append({"trace": go.Scatter(
            x=dag["Date"], y=_get_y(dag, dag_nc, unit), name=cfg["dag_label"],
            line=dict(color=DAG_COLOR, width=2.2, dash="dash"),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{cfg['dag_label']}: %{{y:.1f}}{suffix}<extra></extra>")})

    fig_ts = go.Figure()
    for s in traces:
        fig_ts.add_trace(s["trace"])
    fig_ts.update_layout(
        **_BASE, height=380,
        title=dict(text=f"{cfg['cit_label']}  vs  {cfg['dag_label']}  ·  Net  ·  {ylabel}",
                   font=dict(size=12,color="#333"), x=0),
        margin=dict(l=52,r=20,t=42,b=72),
        legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",font_size=10),
        xaxis=dict(**_ax(x=True),tickformat="%b '%y"),
    )
    fig_ts.update_yaxes(title_text=ylabel, title_font_size=10, **_ax())
    st.plotly_chart(fig_ts, width='stretch')

    # ── 2. Gap bar chart ───────────────────────────────────────────────────────
    if merged is not None and not merged.empty:
        gap = np.asarray(merged["gap"], dtype=float)
        fig_gap = go.Figure(go.Bar(
            x=merged["Date"], y=gap,
            marker=dict(color=[C_LONG if v>=0 else C_SHORT for v in gap],
                        opacity=0.82, line=dict(width=0)),
            hovertemplate="<b>%{x|%b %Y}</b><br>CIT−Disagg: %{y:+.1f}k<extra></extra>",
        ))
        fig_gap.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.14)")
        fig_gap.update_layout(
            **_BASE, height=260,
            title=dict(text=f"Gap: {cfg['cit_label']} minus {cfg['dag_label']}  ·  k lots",
                       font=dict(size=11,color="#444"), x=0),
            margin=dict(l=50,r=12,t=36,b=68), showlegend=False,
            xaxis=dict(**_ax(x=True),tickformat="%b '%y"),
            yaxis=dict(**_ax(),title_text="k lots",title_font_size=10),
        )
        st.plotly_chart(fig_gap, width='stretch')

    # ── 3. Long / Short breakdown ──────────────────────────────────────────────
    cit_lc, cit_sc = cfg.get("cit_long"), cfg.get("cit_short")
    dag_lc, dag_sc = cfg.get("dag_long"), cfg.get("dag_short")
    if cit_lc and dag_lc and cit_lc in cit.columns and dag_lc in dag.columns:
        st.markdown("**Long & Short breakdown**")
        ch1, ch2 = st.columns(2)
        for ch, (cit_col, dag_col, lbl, clr) in zip(
            [ch1, ch2],
            [(cit_lc, dag_lc, "Long", C_LONG), (cit_sc, dag_sc, "Short", C_SHORT)]
        ):
            with ch:
                fig_ls = go.Figure()
                fig_ls.add_trace(go.Scatter(
                    x=cit["Date"], y=_get_y(cit, cit_col, unit), name=f"CIT {lbl}",
                    line=dict(color=clr, width=2.0),
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>CIT {lbl}: %{{y:.1f}}{suffix}<extra></extra>"))
                if dag_col and dag_col in dag.columns:
                    fig_ls.add_trace(go.Scatter(
                        x=dag["Date"], y=_get_y(dag, dag_col, unit), name=f"Disagg {lbl}",
                        line=dict(color=clr, width=2.0, dash="dash"),
                        hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Disagg {lbl}: %{{y:.1f}}{suffix}<extra></extra>"))
                fig_ls.update_layout(
                    **_BASE, height=280,
                    title=dict(text=f"{lbl} positions  ·  {ylabel}", font=dict(size=11,color="#444"), x=0),
                    margin=dict(l=50,r=12,t=36,b=68), showlegend=True,
                    legend=dict(orientation="h",y=-0.32,x=0.5,xanchor="center",font_size=10),
                    xaxis=dict(**_ax(x=True),tickformat="%b '%y"),
                    yaxis=dict(**_ax(),title_text=ylabel,title_font_size=10))
                st.plotly_chart(fig_ls, width='stretch')

    # ── 4. Rolling correlation ─────────────────────────────────────────────────
    if merged is not None and len(merged) > 12:
        with st.expander("Rolling 52-week Correlation", expanded=False):
            s_cit = pd.Series(np.asarray(merged["cit_net"],dtype=float), index=merged["Date"])
            s_dag = pd.Series(np.asarray(merged["dag_net"],dtype=float), index=merged["Date"])
            roll  = s_cit.rolling(52).corr(s_dag).dropna()
            fig_rc = go.Figure(go.Scatter(
                x=roll.index, y=roll.values, mode="lines",
                line=dict(color=color, width=2.0),
                fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.08)",
                hovertemplate="<b>%{x|%b %Y}</b><br>Corr: %{y:.2f}<extra></extra>"))
            fig_rc.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.12)")
            fig_rc.update_layout(
                **_BASE, height=270,
                title=dict(text="Rolling 52-week Correlation — CIT vs Disagg",
                           font=dict(size=11,color="#444"), x=0),
                margin=dict(l=50,r=12,t=36,b=50),
                xaxis=dict(**_ax(x=True),tickformat="%b '%y"),
                yaxis=dict(**_ax(),title_text="Correlation",title_font_size=10,range=[-1.1,1.1]))
            st.plotly_chart(fig_rc, width='stretch')

    # ── 5. Data table ──────────────────────────────────────────────────────────
    with st.expander("Data table", expanded=False):
        cols_cit = [c for c in [cit_nc, cfg.get("cit_long"), cfg.get("cit_short")] if c and c in cit.columns]
        cols_dag = [c for c in [dag_nc, cfg.get("dag_long"), cfg.get("dag_short")] if c and c in dag.columns]
        tbl_cit  = cit[["Date"]+cols_cit].rename(columns={c: f"CIT {c}" for c in cols_cit}).sort_values("Date",ascending=False).head(60)
        tbl_dag  = dag[["Date"]+cols_dag].rename(columns={c: f"Dag {c}" for c in cols_dag}).sort_values("Date",ascending=False).head(60)
        tbl = pd.merge(tbl_cit, tbl_dag, on="Date", how="outer").sort_values("Date",ascending=False).reset_index(drop=True)
        tbl["Date"] = pd.to_datetime(tbl["Date"]).dt.strftime("%d %b '%y")
        num_c = [c for c in tbl.columns if c!="Date"]
        styled = (tbl.style.format({c:"{:,.0f}" for c in num_c}, na_rep="—").hide(axis="index"))
        st.dataframe(styled, width='stretch', height=380)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.05rem;font-weight:700;color:#1a56db;"
        "margin-bottom:16px;letter-spacing:.01em'>COMPREHENSIVE COT</div>",
        unsafe_allow_html=True)

    commodity = st.selectbox("Commodity", list(COMM_NAMES.keys()),
                             format_func=lambda x: COMM_NAMES[x], key="sb_commodity")
    color = COMM_COLORS[commodity]

    cit_ok = commodity in CIT_COMMS
    if cit_ok:
        report = st.radio("Report", ["CIT","Disagg"], horizontal=True, key="rb_report")
    else:
        report = "Disagg"
        st.markdown("<div style='font-size:.73rem;color:#999;margin:-6px 0 8px'>"
                    "RC/LCC — Disaggregated only</div>", unsafe_allow_html=True)

    version_key = None
    if report == "Disagg":
        version = st.radio("Version", ["F&O combined","Fut only","Options only"], horizontal=True, key="rb_version")
        version_key = "F&O" if "F&O" in version else ("Opt" if "Options" in version else "Fut")

    st.markdown("---")
    st.markdown("<div style='font-size:.78rem;font-weight:600;color:#444;"
                "margin-bottom:6px'>Date range</div>", unsafe_allow_html=True)
    _max_date = pd.read_parquet(CIT_FILE, columns=["Date"])["Date"].max().date()
    start_date = st.date_input("From", value=datetime.date(2020,1,1),
                               min_value=datetime.date(2010,1,1), max_value=_max_date, key="dt_from")
    end_date   = st.date_input("To",   value=_max_date,
                               min_value=datetime.date(2010,1,1), max_value=_max_date, key="dt_to")



# ══════════════════════════════════════════════════════════════════════════════
# LOAD + FILTER
# ══════════════════════════════════════════════════════════════════════════════
if report == "CIT":
    raw = load_cit()
    df_all_crops = None
    df = raw[
        (raw["Commodity"]==commodity) &
        (raw["Date"]>=pd.Timestamp(start_date)) &
        (raw["Date"]<=pd.Timestamp(end_date))
    ].sort_values("Date").reset_index(drop=True)
else:
    raw = load_options_only() if version_key == "Opt" else load_disagg(version_key)
    df_all_crops = raw[
        (raw["Commodity"]==commodity) &
        (raw["Date"]>=pd.Timestamp(start_date)) &
        (raw["Date"]<=pd.Timestamp(end_date))
    ].sort_values(["Crop","Date"]).reset_index(drop=True)
    df = df_all_crops[df_all_crops["Crop"]=="All"].sort_values("Date").reset_index(drop=True)

is_options = (report == "Disagg" and version_key == "Opt")

# ── Inject Rollex price (replaces static Px with roll-adjusted daily series) ──
df = _inject_rollex(df, commodity)
if df_all_crops is not None:
    df_all_crops = _inject_rollex(df_all_crops, commodity)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
ver_lbl = "" if report=="CIT" else f" — {version}"
st.markdown(
    f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>"
    f"<div style='width:5px;height:38px;background:{color};border-radius:3px'></div>"
    f"<div>"
    f"<div style='font-size:1.2rem;font-weight:700;color:{color}'>{COMM_NAMES[commodity]}</div>"
    f"<div style='font-size:.73rem;color:#888'>{report}{ver_lbl} &nbsp;·&nbsp; "
    f"{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}</div>"
    f"</div></div>", unsafe_allow_html=True)
st.markdown("---")

if df.empty:
    st.warning("No data for the selected filters."); st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TAB — RECAP (CHARTS)
# ══════════════════════════════════════════════════════════════════════════════
def render_recap_charts(d, report, color, commodity):
    if d.empty:
        st.warning("No data for the selected filters."); return

    d   = d.sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(d["Date"])

    def gc(name):
        return d[name].astype(float) if name in d.columns else pd.Series(np.nan, index=d.index)

    # Nominal multiplier
    size = CONTRACT_SIZE.get(commodity, 1)
    unit = CONTRACT_UNIT.get(commodity, "MT")
    ccy  = "GBP" if commodity == "LCC" else "USD"
    px   = gc("Px")
    mult = (px * size / 100 / 1_000_000) if unit == "lbs" else (px * size / 1_000_000)
    oi   = gc("Total OI").replace(0, np.nan)

    def _line(title, series_dict, clrs=None):
        dflt = [C_LONG, C_SHORT, C_NET, "#f59e0b", "#7c3aed"]
        if clrs is None: clrs = dflt
        fig = go.Figure()
        fig.update_layout(
            **_BASE,
            title=dict(text=f"{commodity} — {title}", font=dict(size=10, color="#374151")),
            height=260,
            margin=dict(l=40, r=8, t=36, b=48),
            showlegend=True,
            legend=dict(orientation="h", y=-0.28, font=dict(size=9)),
            xaxis=dict(**_ax(x=True)),
            yaxis=dict(**_ax()),
        )
        for i, (name, y) in enumerate(series_dict.items()):
            fig.add_trace(go.Scatter(
                x=dates, y=y, name=name,
                line=dict(color=clrs[i % len(clrs)], width=1.5)
            ))
        return fig

    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)

    if report == "CIT":
        spec_net  = gc("Spec Net")
        idx_net   = gc("Index Net")
        ls_long   = (gc("Spec Long") + gc("Non Rep Long")) / 1000
        ls_short  = (gc("Spec Short") + gc("Non Rep Short")) / 1000

        with c1:
            st.plotly_chart(_line(
                f"Nominal Exposure M {ccy}",
                {"Net Spec": spec_net * mult, "Net Index": idx_net * mult},
                [C_NET, C_LONG]
            ), width='stretch')
        with c2:
            st.plotly_chart(_line(
                f"Commercial Nominal M {ccy}",
                {"Gross Long": gc("Comm Long") * mult, "Gross Short": gc("Comm Short") * mult},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c3:
            st.plotly_chart(_line(
                "Gross % of OI",
                {"Lrg+Sml Long %":  (gc("Spec Long")  + gc("Non Rep Long"))  / oi * 100,
                 "Lrg+Sml Short %": (gc("Spec Short") + gc("Non Rep Short")) / oi * 100},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c4:
            st.plotly_chart(_line(
                "# of Traders",
                {"Large Long": gc("Traders Spec Long"), "Large Short": gc("Traders Spec Short")},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c5:
            st.plotly_chart(_line(
                "Spec Gross k lots",
                {"Large Long": gc("Spec Long") / 1000, "Large Short": gc("Spec Short") / 1000},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c6:
            st.plotly_chart(_line(
                "Commercial Gross k lots",
                {"Comm Long": gc("Comm Long") / 1000, "Comm Short": gc("Comm Short") / 1000},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c7:
            st.plotly_chart(_line(
                "Net Spec & Net Index k lots",
                {"Net Spec": spec_net / 1000, "Net Index": idx_net / 1000},
                [C_NET, C_LONG]
            ), width='stretch')
        with c8:
            st.plotly_chart(_line(
                "Large+Small k lots",
                {"L+S Long": ls_long, "L+S Short": ls_short},
                [C_LONG, C_SHORT]
            ), width='stretch')

    else:  # Disagg
        mm_net   = gc("MM Net")
        swap_net = gc("Swap Net")
        mm_all_l = (gc("MM Long") + gc("Other Long")) / 1000
        mm_all_s = (gc("MM Short") + gc("Other Short")) / 1000

        with c1:
            st.plotly_chart(_line(
                f"MM Nominal M {ccy}",
                {"MM Net": mm_net * mult, "Swap Net": swap_net * mult},
                [C_NET, C_LONG]
            ), width='stretch')
        with c2:
            st.plotly_chart(_line(
                f"Commercial Nominal M {ccy}",
                {"Prod Long": gc("Producer Long") * mult, "Prod Short": gc("Producer Short") * mult},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c3:
            st.plotly_chart(_line(
                "MM Gross % of OI",
                {"MM Long %":  gc("MM Long")  / oi * 100,
                 "MM Short %": gc("MM Short") / oi * 100},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c4:
            st.plotly_chart(_line(
                "# of Traders",
                {"MM Long": gc("Traders MM Long"), "MM Short": gc("Traders MM Short")},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c5:
            st.plotly_chart(_line(
                "MM Gross k lots",
                {"MM Long": gc("MM Long") / 1000, "MM Short": gc("MM Short") / 1000},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c6:
            st.plotly_chart(_line(
                "Commercial k lots",
                {"Prod Long": gc("Producer Long") / 1000, "Prod Short": gc("Producer Short") / 1000},
                [C_LONG, C_SHORT]
            ), width='stretch')
        with c7:
            st.plotly_chart(_line(
                "MM Net & Swap Net k lots",
                {"MM Net": mm_net / 1000, "Swap Net": swap_net / 1000},
                [C_NET, C_LONG]
            ), width='stretch')
        with c8:
            st.plotly_chart(_line(
                "MM+Other k lots",
                {"MM+Other Long": mm_all_l, "MM+Other Short": mm_all_s},
                [C_LONG, C_SHORT]
            ), width='stretch')

    # ── Roll Yield vs Positioning ──────────────────────────────────────────────
    with st.expander("Roll Yield vs Positioning", expanded=True):
        ry_all = load_roll_yield()
        ry_comm = ry_all[ry_all["Commodity"] == commodity].copy()
        if ry_comm.empty:
            st.info("Roll yield data not available for this commodity.")
        else:
            cot_sorted = d.sort_values("Date").copy()
            ry_sorted  = ry_comm.sort_values("Date")
            merged_ry  = pd.merge_asof(cot_sorted, ry_sorted[["Date","roll_yield_pct"]], on="Date", direction="backward")

            ry_vals = merged_ry["roll_yield_pct"].values.astype(float)

            if report == "CIT":
                short_vals = ((gc("Spec Short") + gc("Non Rep Short")) / 1000).values.astype(float)
                net_vals   = (gc("Spec Net") / 1000).values.astype(float)
                short_lbl  = "Spec + Non-Rep Short (k lots)"
                net_lbl    = "Spec Net (k lots)"
            else:
                short_vals = ((gc("MM Short") + gc("Other Short") + gc("Non Rep Short")) / 1000).values.astype(float)
                net_vals   = (gc("MM Net") / 1000).values.astype(float)
                short_lbl  = "MM + Other + Non-Rep Short (k lots)"
                net_lbl    = "MM Net (k lots)"

            def _ry_scatter(x, y, ylabel, title):
                mask = ~(np.isnan(x) | np.isnan(y))
                xm, ym = x[mask], y[mask]
                if len(xm) < 5:
                    return go.Figure()
                # polynomial trendline degree 2
                coeffs = np.polyfit(xm, ym, 2)
                x_line = np.linspace(xm.min(), xm.max(), 200)
                y_line = np.polyval(coeffs, x_line)
                # R²
                y_hat  = np.polyval(coeffs, xm)
                ss_res = np.sum((ym - y_hat) ** 2)
                ss_tot = np.sum((ym - ym.mean()) ** 2)
                r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                fig = go.Figure()
                # all points
                fig.add_trace(go.Scatter(
                    x=xm, y=ym, mode="markers",
                    marker=dict(color=color, size=6, opacity=0.65, line=dict(width=0.4, color="white")),
                    hovertemplate="Roll Yield: %{x:.1f}%<br>" + ylabel.split(" (")[0] + ": %{y:.1f}k<extra></extra>",
                    showlegend=False,
                ))
                # trendline
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    showlegend=False,
                ))
                # latest point
                fig.add_trace(go.Scatter(
                    x=[x[~np.isnan(x) & ~np.isnan(y)][-1]],
                    y=[y[~np.isnan(x) & ~np.isnan(y)][-1]],
                    mode="markers",
                    marker=dict(color="#f97316", size=10, symbol="circle", line=dict(width=1.5, color="white")),
                    showlegend=False,
                    hovertemplate="<b>Latest</b><br>Roll Yield: %{x:.1f}%<br>" + ylabel.split(" (")[0] + ": %{y:.1f}k<extra></extra>",
                ))
                fig.add_annotation(
                    x=0.98, y=0.98, xref="paper", yref="paper",
                    text=f"R² = {r2:.4f}", showarrow=False,
                    font=dict(size=13, color="#111"), bgcolor="rgba(255,255,255,0.7)",
                    borderpad=4, xanchor="right", yanchor="top",
                )
                fig.update_layout(
                    **_BASE,
                    title=dict(text=f"{commodity} — {title}", font=dict(size=11, color="#374151")),
                    height=380,
                    margin=dict(l=48, r=16, t=44, b=48),
                    xaxis=dict(**_ax(), title=dict(text="Roll Yield 1yr (%)", font=dict(size=9)), tickformat=".1f", ticksuffix="%"),
                    yaxis=dict(**_ax(), title=dict(text=ylabel, font=dict(size=9))),
                    showlegend=False,
                )
                return fig

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(_ry_scatter(ry_vals, short_vals, short_lbl, "Large Spec Short vs Roll Yield"), width='stretch')
            with col_b:
                st.plotly_chart(_ry_scatter(ry_vals, net_vals, net_lbl, "Net Spec vs Roll Yield"), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# SPEC VAR TAB
# ══════════════════════════════════════════════════════════════════════════════
def render_spec_var(commodity: str, df_cot: pd.DataFrame, report: str, color: str,
                    start_date=None, end_date=None):
    var_df = _build_var_df(commodity)
    lot    = VAR_LOT_USD.get(commodity, 10)
    t0 = pd.Timestamp(start_date) if start_date is not None else var_df["Date"].min() if not var_df.empty else pd.Timestamp("2000-01-01")
    t1 = pd.Timestamp(end_date)   if end_date   is not None else var_df["Date"].max() if not var_df.empty else pd.Timestamp.now()
    if not var_df.empty:
        var_df = var_df[(var_df["Date"] >= t0) & (var_df["Date"] <= t1)]

    # colour palette for cross-commodity (KC/RC = blue family, CC/LCC = red/orange family)
    _VAR_COLORS = {
        "KC": "#1d4ed8", "RC": "#38bdf8",
        "CC": "#b91c1c", "LCC":"#fb923c",
        "SB": "#059669", "CT": "#7c3aed",
    }
    _VAR_DASH = {
        "KC": "solid",   "RC": "dash",
        "CC": "solid",   "LCC":"dash",
        "SB": "solid",   "CT": "solid",
    }
    _VAR_WIDTH = {
        "KC": 2.0, "RC": 1.6,
        "CC": 2.0, "LCC":1.6,
        "SB": 1.6, "CT": 1.6,
    }

    # ── derive MM+Other columns ───────────────────────────────────────────────
    df_c = df_cot.copy()
    for side in ["Net", "Long", "Short"]:
        col = f"MM + Other {side}"
        if col not in df_c.columns:
            mm_c, ot_c = f"MM {side}", f"Other {side}"
            if mm_c in df_c.columns and ot_c in df_c.columns:
                df_c[col] = df_c[mm_c] + df_c[ot_c]

    # ── Net-only spec options ─────────────────────────────────────────────────
    preferred_net = [
        "MM Net", "MM + Other Net", "Other Net",
        "Spec Net", "Index Net", "Non Rep Net",
        "Combined Spec Net", "Swap Net",
    ]
    spec_opts = [c for c in preferred_net if c in df_c.columns]
    if not spec_opts:
        st.warning("No spec Net columns found for this report type.")
        return

    # ── controls ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        spec_sel = st.selectbox("Spec Position", spec_opts, key=f"var_spec_{commodity}")
    with c2:
        win_sel = st.radio("Vol Window", [20, 60, 120], horizontal=True,
                           format_func=lambda x: f"{x}D", key=f"var_win_{commodity}")

    if var_df.empty:
        st.warning(f"No rollex / VaR data for {commodity}.")
        return

    base_name = spec_sel.rsplit(" ", 1)[0]
    long_col  = f"{base_name} Long"
    short_col = f"{base_name} Short"
    _ver_lbl  = f"{report} {version_key}" if report == "Disagg" and version_key else report

    # ── Expander 2: Spec Book VaR — Net/Long/Short + WoW change ──────────────
    with st.expander(f"Spec Book VaR — {spec_sel}  ·  {win_sel}D  [{_ver_lbl}]", expanded=True):
        vcol = f"vol_{win_sel}"
        if vcol not in var_df.columns or spec_sel not in df_c.columns:
            st.info("Data not available.")
        else:
            vol_s = var_df[["Date", vcol]].dropna().copy()
            extra_cols = [c for c in [long_col, short_col] if c in df_c.columns]
            cols_sel = ["Date", spec_sel, "Px"] + extra_cols
            df_m = pd.merge_asof(
                df_c.sort_values("Date")[cols_sel].dropna(subset=[spec_sel]),
                vol_s.sort_values("Date"),
                on="Date", direction="backward",
            ).dropna(subset=[vcol])
            df_m["vpl"] = df_m["Px"] * lot * df_m[vcol] * _CONF_Z

            df_m = df_m.sort_values("Date")
            df_m["Net VaR ($M)"]   = df_m[spec_sel] * df_m["vpl"] / 1e6
            if long_col  in df_m.columns: df_m["Long VaR ($M)"]  = df_m[long_col]  * df_m["vpl"] / 1e6
            if short_col in df_m.columns: df_m["Short VaR ($M)"] = df_m[short_col] * df_m["vpl"] / 1e6
            df_m["Δ WoW"] = df_m["Net VaR ($M)"].diff()
            df_m = df_m.sort_values("Date", ascending=False).head(52)

            def _mfmt(v, invert=False):
                if pd.isna(v): return "—"
                pos = (v > 0) if not invert else (v < 0)
                cls = "rpos" if pos else "rneg"
                return f"<span class='{cls}'>${v:.1f}M</span>"
            def _dfmt(v):
                if pd.isna(v): return "—"
                cls = "rpos" if v > 0 else "rneg"
                arrow = "▲" if v > 0 else "▼"
                return f"<span class='{cls}'>{arrow}${abs(v):.1f}M</span>"

            rows_e2 = ""
            for _, r in df_m.iterrows():
                rows_e2 += (
                    f"<tr><td class='idx'>{r['Date'].strftime('%d-%b-%y')}</td>"
                    f"<td>{_mfmt(r.get('Net VaR ($M)'))}</td>"
                    f"<td>{_dfmt(r.get('Δ WoW'))}</td>"
                    f"<td>{_mfmt(r.get('Long VaR ($M)'))}</td>"
                    f"<td>{_mfmt(r.get('Short VaR ($M)'), invert=True)}</td></tr>"
                )
            hdr_e2 = ("<tr style='background:#f3f4f6'>"
                      "<th class='idx'>Date</th>"
                      "<th>Net VaR ($M)</th><th>Δ WoW</th>"
                      "<th>Long VaR ($M)</th><th>Short VaR ($M)</th></tr>")
            st.markdown(
                f"{_RECAP_CSS}<div style='overflow-x:auto;overflow-y:auto;max-height:480px'>"
                f"<table class='rtbl'><thead>{hdr_e2}</thead><tbody>{rows_e2}</tbody></table></div>",
                unsafe_allow_html=True,
            )

    # ── Expander 3: Book VaR timeseries — selected window only ───────────────
    with st.expander(f"Book VaR — {win_sel}D timeseries  ({spec_sel}) [{_ver_lbl}]", expanded=True):
        vcol = f"vol_{win_sel}"
        if spec_sel not in df_c.columns or vcol not in var_df.columns:
            st.info("Data not available.")
        else:
            vol_s = var_df[["Date", vcol]].dropna().copy()
            dm = pd.merge_asof(
                df_c.sort_values("Date")[["Date", spec_sel, "Px"]].dropna(subset=[spec_sel]),
                vol_s.sort_values("Date"), on="Date", direction="backward",
            ).dropna(subset=[vcol])
            dm["vpl"] = dm["Px"] * lot * dm[vcol] * _CONF_Z
            dm["VaR"] = dm[spec_sel] * dm["vpl"] / 1e6
            dm = dm.sort_values("Date")

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=dm["Date"], y=dm["VaR"],
                mode="lines", name=f"{win_sel}D Net VaR ($M)",
                line=dict(color=color, width=1.8),
            ))
            fig3.update_layout(
                **_BASE, height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
                yaxis=dict(**_ax(), title=dict(text="Net VaR ($M)", font=dict(size=9))),
                xaxis=dict(**_ax(x=True)),
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Expander 4: Long / Short VaR decomposition ────────────────────────────
    with st.expander(f"Long / Short VaR — {win_sel}D  ({base_name}) [{_ver_lbl}]", expanded=False):
        vcol = f"vol_{win_sel}"
        has_legs = any(c in df_c.columns for c in [long_col, short_col])
        if not has_legs or vcol not in var_df.columns:
            st.info("Long / Short columns not available for this report type.")
        else:
            vol_ls = var_df[["Date", vcol]].dropna().copy()
            ls_cols = ["Date", "Px"] + [c for c in [long_col, short_col] if c in df_c.columns]
            dm_ls = pd.merge_asof(
                df_c.sort_values("Date")[ls_cols],
                vol_ls.sort_values("Date"), on="Date", direction="backward",
            ).dropna(subset=[vcol]).sort_values("Date")
            dm_ls["vpl"] = dm_ls["Px"] * lot * dm_ls[vcol] * _CONF_Z
            if long_col  in dm_ls.columns: dm_ls["Long VaR ($M)"]  = dm_ls[long_col]  * dm_ls["vpl"] / 1e6
            if short_col in dm_ls.columns: dm_ls["Short VaR ($M)"] = dm_ls[short_col] * dm_ls["vpl"] / 1e6

            fig4 = go.Figure()
            if "Long VaR ($M)" in dm_ls.columns:
                fig4.add_trace(go.Scatter(
                    x=dm_ls["Date"], y=dm_ls["Long VaR ($M)"],
                    mode="lines", name="Long VaR ($M)",
                    line=dict(color=C_LONG, width=1.6),
                ))
            if "Short VaR ($M)" in dm_ls.columns:
                fig4.add_trace(go.Scatter(
                    x=dm_ls["Date"], y=dm_ls["Short VaR ($M)"],
                    mode="lines", name="Short VaR ($M)",
                    line=dict(color=C_SHORT, width=1.6),
                ))
            fig4.update_layout(
                **_BASE, height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
                yaxis=dict(**_ax(), title=dict(text="VaR ($M)", font=dict(size=9))),
                xaxis=dict(**_ax(x=True)),
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ── Expander 5: Cross-Commodity Comparison ────────────────────────────────
    with st.expander("Cross-Commodity Comparison", expanded=False):
        c_win = st.radio(
            "Window", [20, 60, 120], horizontal=True,
            format_func=lambda x: f"{x}D", key=f"var_cross_win_{commodity}",
        )

        # helper: build NetVaR timeseries for one commodity + one spec column
        def _cross_ts(comm, df_source, spec_col):
            sub = df_source[df_source["Commodity"] == comm].copy()
            if "Crop" in sub.columns:
                sub = sub[sub["Crop"] == "All"]
            if sub.empty or spec_col not in sub.columns:
                return None, None
            cv_df = _build_var_df(comm)
            if cv_df.empty:
                return None, None
            vc = f"vol_{c_win}"
            if vc not in cv_df.columns:
                return None, None
            vol_s2 = cv_df[["Date", vc]].dropna().copy()
            mc = pd.merge_asof(
                sub.sort_values("Date")[["Date", spec_col, "Px"]].dropna(subset=[spec_col]),
                vol_s2.sort_values("Date"), on="Date", direction="backward",
            ).dropna(subset=[vc])
            mc["vpl"] = mc["Px"] * VAR_LOT_USD.get(comm, 10) * mc[vc] * _CONF_Z
            mc = mc[(mc["Date"] >= t0) & (mc["Date"] <= t1)]
            if mc.empty:
                return None, None
            mc["NetVaR"] = mc[spec_col] * mc["vpl"] / 1e6
            snap = mc.sort_values("Date").iloc[-1]
            snap_info = {
                "name": COMM_NAMES[comm],
                "date": snap["Date"].strftime("%d-%b-%y"),
                "pos_k": f"{snap[spec_col]/1000:.1f}k",
                "vpl": f"${snap['vpl']:,.0f}",
                "net_var": snap["NetVaR"],
            }
            return mc[["Date","NetVaR"]].copy(), snap_info

        def _render_cross_section(comms, df_source, spec_opts_list, sel_key, chart_key):
            c_spec = st.selectbox("Spec", spec_opts_list, key=sel_key)
            # derive MM+Other if needed
            df_src = df_source.copy()
            if "MM + Other" in c_spec:
                side = c_spec.split()[-1]
                df_src["MM + Other Net"] = (
                    df_src.get("MM Net", pd.Series(0, index=df_src.index)) +
                    df_src.get("Other Net", pd.Series(0, index=df_src.index))
                )
            snaps, fig_x = [], go.Figure()
            for comm in comms:
                ts, snap = _cross_ts(comm, df_src, c_spec)
                if ts is None:
                    continue
                snaps.append(snap)
                ts = ts.sort_values("Date").copy()
                fig_x.add_trace(go.Scatter(
                    x=ts["Date"], y=ts["NetVaR"],
                    mode="lines", name=COMM_NAMES[comm],
                    line=dict(
                        color=_VAR_COLORS.get(comm, "#888"),
                        width=_VAR_WIDTH.get(comm, 1.5),
                        dash=_VAR_DASH.get(comm, "solid"),
                    ),
                ))
            if snaps:
                snaps.sort(key=lambda x: abs(x["net_var"]) if pd.notna(x["net_var"]) else 0, reverse=True)
                sr = ""
                for r in snaps:
                    nv = r["net_var"]
                    nv_cls = "rpos" if nv > 0 else "rneg"
                    nv_s = f"<span class='{nv_cls}'>${nv:.1f}M</span>" if pd.notna(nv) else "—"
                    sr += (f"<tr><td class='idx'>{r['name']}</td><td>{r['date']}</td>"
                           f"<td>{r['pos_k']}</td><td>{r['vpl']}</td><td>{nv_s}</td></tr>")
                hdr_x = ("<tr style='background:#f3f4f6'><th class='idx'>Commodity</th>"
                         f"<th>Date</th><th>{c_spec} (k)</th>"
                         "<th>VaR/lot($)</th><th>Net VaR($M)</th></tr>")
                st.markdown(
                    f"{_RECAP_CSS}<div style='overflow-x:auto'><table class='rtbl'>"
                    f"<thead>{hdr_x}</thead><tbody>{sr}</tbody></table></div>",
                    unsafe_allow_html=True,
                )
            fig_x.update_layout(
                **_BASE, height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                yaxis=dict(**_ax(), title=dict(text="Net VaR ($M)", font=dict(size=9))),
                xaxis=dict(**_ax(x=True)),
            )
            st.plotly_chart(fig_x, use_container_width=True, key=chart_key)

        # ── Section A: Disagg F&O — all 6 commodities ─────────────────────────
        st.markdown("**Disagg F&O — all commodities**  ·  <span style='font-size:.75rem;color:#888'>always Disagg F&O regardless of sidebar selection</span>", unsafe_allow_html=True)
        disagg_fo = load_disagg("F&O")
        _render_cross_section(
            list(COMM_NAMES.keys()), disagg_fo,
            ["MM Net", "MM + Other Net", "Combined Spec Net"],
            f"var_cross_disagg_{commodity}",
            f"var_cross_chart_disagg_{commodity}",
        )

        st.markdown("---")

        # ── Section B: CIT — US markets only ─────────────────────────────────
        st.markdown("**CIT — US markets (KC · CC · SB · CT)**  ·  <span style='font-size:.75rem;color:#888'>always CIT regardless of sidebar selection</span>", unsafe_allow_html=True)
        cit_all = load_cit()
        _render_cross_section(
            ["KC", "CC", "SB", "CT"], cit_all,
            ["Spec Net", "Index Net", "Non Rep Net", "Combined Spec Net"],
            f"var_cross_cit_{commodity}",
            f"var_cross_chart_cit_{commodity}",
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAIRS TAB  (KC+RC  and  CC+LCC  — always Disagg F&O)
# ══════════════════════════════════════════════════════════════════════════════
def render_pairs(start_date=None, end_date=None):
    st.markdown(
        "<div style='font-size:.75rem;color:#555;margin-bottom:14px;padding:7px 14px;"
        "background:#f8f9fb;border-radius:6px;border:1px solid #e5e7eb'>"
        "Always uses <b>Disaggregated F&O</b> data regardless of sidebar report selection. "
        "Compares equivalent trader classes across both legs of each market pair.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 3])
    with c1:
        pair = st.radio("Pair", ["KC + RC", "CC + LCC"], horizontal=True, key="pair_sel")
    with c2:
        view = st.radio("View", ["Combined Net", "Individual Legs"], horizontal=True, key="pair_view")

    comm_a, comm_b = ("KC", "RC") if "KC" in pair else ("CC", "LCC")
    clr_a = COMM_COLORS[comm_a]
    clr_b = COMM_COLORS[comm_b]

    fo   = load_disagg("F&O")
    t0   = pd.Timestamp(start_date) if start_date else pd.Timestamp("2010-01-01")
    t1   = pd.Timestamp(end_date)   if end_date   else pd.Timestamp.now()

    def _get(comm):
        sub = fo[(fo["Commodity"] == comm) & (fo["Crop"] == "All")].sort_values("Date").reset_index(drop=True)
        return sub[(sub["Date"] >= t0) & (sub["Date"] <= t1)].reset_index(drop=True)

    da = _get(comm_a)
    db = _get(comm_b)

    if da.empty or db.empty:
        st.warning("Not enough data for this pair in the selected date range.")
        return

    suf_a, suf_b = f"_{comm_a}", f"_{comm_b}"
    mg = pd.merge(da, db, on="Date", suffixes=(suf_a, suf_b), how="inner")
    if mg.empty:
        st.warning("No overlapping dates for this pair.")
        return

    ALL_LEGS = ["MM Net", "MM Long", "MM Short",
                "Comm Net", "Comm Long", "Comm Short",
                "Other Net", "Non Rep Net", "Combined Spec Net"]
    NET_OPTS = [c for c in ALL_LEGS if f"{c}{suf_a}" in mg.columns]
    LEG_OPTS = NET_OPTS  # same list for individual view

    def _col(comm, base):
        key = f"{base}_{comm}"
        return mg[key] if key in mg.columns else pd.Series(np.nan, index=mg.index)

    st.markdown("---")

    # ── COMBINED NET ──────────────────────────────────────────────────────────
    if view == "Combined Net":
        defaults = [n for n in ["MM Net", "Comm Net"] if n in NET_OPTS]
        sel_nets = st.multiselect(
            "COT Elements", NET_OPTS, default=defaults, key="pair_nets",
        )
        if not sel_nets:
            st.info("Select at least one element above.")
            return

        palette = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#d97706"]
        fig = go.Figure()
        for i, net in enumerate(sel_nets):
            sa = _col(comm_a, net)
            sb = _col(comm_b, net)
            combined = sa.fillna(0) + sb.fillna(0)
            clr = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=mg["Date"], y=combined, mode="lines",
                name=f"{net} (combined)",
                line=dict(color=clr, width=2.0),
            ))
            fig.add_trace(go.Scatter(
                x=mg["Date"], y=sa, mode="lines",
                name=f"{net} ({comm_a})",
                line=dict(color=clr, width=1.2, dash="dot"),
                visible="legendonly",
            ))
            fig.add_trace(go.Scatter(
                x=mg["Date"], y=sb, mode="lines",
                name=f"{net} ({comm_b})",
                line=dict(color=clr, width=1.2, dash="dash"),
                visible="legendonly",
            ))
        fig.update_layout(
            **_BASE, height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
            yaxis=dict(**_ax(), title=dict(text="k lots", font=dict(size=9))),
            xaxis=dict(**_ax(x=True)),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"pair_net_{comm_a}{comm_b}")

    # ── INDIVIDUAL LEGS ───────────────────────────────────────────────────────
    else:
        sel_leg = st.selectbox("COT Element", LEG_OPTS, key="pair_leg")

        sa = _col(comm_a, sel_leg)
        sb = _col(comm_b, sel_leg)

        sa_chg = sa.diff().dropna()
        sb_chg = sb.diff().dropna()
        shared = sa_chg.index.intersection(sb_chg.index)
        r_val  = None
        if len(shared) > 5:
            r_val = float(np.corrcoef(sa_chg.loc[shared].values, sb_chg.loc[shared].values)[0, 1])

        # Dual-axis timeseries
        st.markdown(
            f"<div style='font-size:.78rem;font-weight:600;color:#374151;margin-bottom:4px'>"
            f"{sel_leg}  —  {COMM_NAMES[comm_a]}  vs  {COMM_NAMES[comm_b]}</div>",
            unsafe_allow_html=True,
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=mg["Date"], y=sa, mode="lines",
            name=comm_a,
            line=dict(color=clr_a, width=1.9),
        ))
        fig2.add_trace(go.Scatter(
            x=mg["Date"], y=sb, mode="lines",
            name=comm_b,
            line=dict(color=clr_b, width=1.9),
        ))
        fig2.update_layout(
            **_BASE, height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
            yaxis=dict(**_ax(), title=dict(text="k lots", font=dict(size=9))),
            xaxis=dict(**_ax(x=True)),
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"pair_leg_{comm_a}{comm_b}")

        # Correlation scatter
        if len(shared) > 5:
            xa = sa_chg.loc[shared].values
            xb = sb_chg.loc[shared].values
            dates_s = mg.loc[shared, "Date"] if "Date" in mg.columns else pd.Series(shared)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=xa, y=xb, mode="markers",
                marker=dict(
                    color=list(range(len(xa))), colorscale="Blues",
                    size=6, opacity=0.75, showscale=False,
                ),
                text=[d.strftime("%d-%b-%y") if hasattr(d, "strftime") else str(d) for d in dates_s],
                hovertemplate=(f"{comm_a} delta: %{{x:.1f}}k<br>"
                               f"{comm_b} delta: %{{y:.1f}}k<br>%{{text}}<extra></extra>"),
                showlegend=False,
            ))
            fig3.add_trace(go.Scatter(
                x=[xa[-1]], y=[xb[-1]], mode="markers",
                marker=dict(color="#f97316", size=10, symbol="star"),
                showlegend=False,
            ))
            fig3.update_layout(
                **_BASE, height=320,
                margin=dict(l=48, r=16, t=16, b=48),
                xaxis=dict(**_ax(), title=dict(text=f"{comm_a} weekly change (k lots)", font=dict(size=9))),
                yaxis=dict(**_ax(), title=dict(text=f"{comm_b} weekly change (k lots)", font=dict(size=9))),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True, key=f"pair_scatter_{comm_a}{comm_b}")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
# Options Only: hide Spreading (options spreads ≠ calendar spreads) and CIT vs Disagg (CIT is fut-only)
if is_options:
    TAB_LABELS = ["Recap","Recap (Charts)","Spec","Commercial","Old / New",
                  "Concentration","Scatter Plot","Spec VaR","Pairs"]
elif report == "CIT":
    TAB_LABELS = ["Recap","Recap (Charts)","Spec","Commercial",
                  "Scatter Plot","CIT vs Disagg","Spec VaR","Pairs"]
else:
    TAB_LABELS = ["Recap","Recap (Charts)","Spec","Commercial","Spreading","Old / New",
                  "Concentration","Scatter Plot","CIT vs Disagg","Spec VaR","Pairs"]

tabs = st.tabs(TAB_LABELS)

if is_options:
    with tabs[0]:  render_recap(df, report, color, commodity, is_options=True)
    with tabs[1]:  render_recap_charts(df, report, color, commodity)
    with tabs[2]:  render_spec(df, report, color)
    with tabs[3]:  render_commercial(df, report, color)
    with tabs[4]:
        if df_all_crops is not None: render_old_new(df_all_crops, color)
        else: st.info("Old / New crop split not available.")
    with tabs[5]:  render_concentration(df, color)
    with tabs[6]:  render_analysis(df, report, color)
    with tabs[7]:  render_spec_var(commodity, df, report, color, start_date, end_date)
    with tabs[8]:  render_pairs(start_date, end_date)
elif report == "CIT":
    with tabs[0]:  render_recap(df, report, color, commodity)
    with tabs[1]:  render_recap_charts(df, report, color, commodity)
    with tabs[2]:  render_spec(df, report, color)
    with tabs[3]:  render_commercial(df, report, color)
    with tabs[4]:  render_analysis(df, report, color)
    with tabs[5]:  render_comparison(commodity, start_date, end_date, color)
    with tabs[6]:  render_spec_var(commodity, df, report, color, start_date, end_date)
    with tabs[7]:  render_pairs(start_date, end_date)
else:
    with tabs[0]:  render_recap(df, report, color, commodity)
    with tabs[1]:  render_recap_charts(df, report, color, commodity)
    with tabs[2]:  render_spec(df, report, color)
    with tabs[3]:  render_commercial(df, report, color)
    with tabs[4]:  render_spreading(df, color)
    with tabs[5]:
        if df_all_crops is not None: render_old_new(df_all_crops, color)
        else: st.info("Old / New crop split not available.")
    with tabs[6]:  render_concentration(df, color)
    with tabs[7]:  render_analysis(df, report, color)
    with tabs[8]:  render_comparison(commodity, start_date, end_date, color)
    with tabs[9]:  render_spec_var(commodity, df, report, color, start_date, end_date)
    with tabs[10]: render_pairs(start_date, end_date)