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
st.set_page_config(page_title="ICEBREAKER — COT", layout="wide",
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
DB_DIR   = Path(__file__).resolve().parent.parent / "Database"
CIT_FILE = DB_DIR / "cot_cit.parquet"
FO_FILE  = DB_DIR / "cot_disagg_futopt.parquet"
FUT_FILE = DB_DIR / "cot_disagg_fut.parquet"

# ── Commodity config ──────────────────────────────────────────────────────────
COMM_COLORS = {
    "KC":"#1a56db","CC":"#d97706","SB":"#059669",
    "CT":"#7c3aed","RC":"#dc2626","LCC":"#0891b2",
}
COMM_NAMES = {
    "KC":"KC — Arabica Coffee","CC":"CC — NYC Cocoa",
    "SB":"SB — Sugar #11","CT":"CT — Cotton #2",
    "RC":"RC — Robusta Coffee","LCC":"LCC — London Cocoa",
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

def show_table(d: pd.DataFrame, pos_cols: list, chg_cols: list, label: str, n=60):
    with st.expander(label, expanded=False):
        avail_p = [c for c in pos_cols if c in d.columns]
        avail_c = [c for c in chg_cols if c in d.columns]
        tbl = d[["Date"] + avail_p].copy().sort_values("Date", ascending=False).head(n)
        tbl["Date"] = pd.to_datetime(tbl["Date"]).dt.strftime("%d %b '%y")
        for c in avail_c:
            tbl[f"Δ {c}"] = d[c].diff().reindex(tbl.index)
        tbl = tbl.reset_index(drop=True)
        num_cols = [c for c in tbl.columns if c != "Date"]

        def _style(s):
            styles = []
            for v in s:
                try:
                    fv = float(v)
                    if fv > 0: styles.append("color:#16a34a")
                    elif fv < 0: styles.append("color:#dc2626")
                    else: styles.append("")
                except: styles.append("")
            return styles

        fmt = {c: "{:,.0f}" for c in num_cols if "Pct" not in c and "%" not in c}
        fmt.update({c: "{:.2f}%" for c in num_cols if "Pct" in c or c == "Px"})
        styled = (tbl.style
                  .format(fmt, na_rep="—")
                  .apply(_style, subset=[c for c in num_cols if c.startswith("Δ")])
                  .hide(axis="index"))
        st.dataframe(styled, use_container_width=True, height=380)


# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _add_price(fig, d, secondary_y=True):
    if "Px" not in d.columns or d["Px"].isna().all(): return
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d["Px"], name="Price",
        line=dict(color=C_PRICE, width=1.2, dash="dot"), opacity=0.65,
        hovertemplate="<b>%{x|%b %Y}</b><br>Price: %{y:.2f}<extra></extra>",
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
    fig.update_yaxes(title_text="Price", title_font_size=10, secondary_y=True,
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SPEC
# ══════════════════════════════════════════════════════════════════════════════
CIT_SPEC = {
    "Large Spec":    {"long":"Spec Long",    "short":"Spec Short",    "net":"Spec Net",    "spread":None},
    "Non-Rep":       {"long":"Non Rep Long", "short":"Non Rep Short", "net":"Non Rep Net", "spread":None},
    "Index Traders": {"long":"Index Long",   "short":"Index Short",   "net":"Index Net",   "spread":None},
    "Combined Spec": {"long":"Combined Spec Long","short":"Combined Spec Short","net":"Combined Spec Net","spread":None},
}
DISAGG_SPEC = {
    "Managed Money":    {"long":"MM Long",    "short":"MM Short",    "net":"MM Net",    "spread":"MM Spread"},
    "Other Rept":       {"long":"Other Long", "short":"Other Short", "net":"Other Net", "spread":"Other Spread"},
    "Non-Rep":          {"long":"Non Rep Long","short":"Non Rep Short","net":"Non Rep Net","spread":None},
    "Swap Dealers":     {"long":"Swap Long",  "short":"Swap Short",  "net":"Swap Net",  "spread":"Swap Spread"},
    "Combined Spec":    {"long":"Combined Spec Long","short":"Combined Spec Short","net":"Combined Spec Net","spread":None},
}

def render_spec(d, report, color):
    cats = CIT_SPEC if report == "CIT" else DISAGG_SPEC
    c1,c2,c3 = st.columns([2,2,1])
    with c1: cat  = st.selectbox("Category", list(cats.keys()), key="spec_cat")
    with c2: view = st.radio("View", ["Net","Legs (Long + Short)"], horizontal=True, key="spec_view")
    with c3: unit = st.radio("Unit", ["k lots","% of OI"], horizontal=True, key="spec_unit")

    cfg = cats[cat]
    lc, sc, nc, spc = cfg["long"], cfg["short"], cfg["net"], cfg["spread"]

    latest = d.iloc[-1]
    prev   = d.iloc[-2] if len(d)>1 else d.iloc[-1]
    pxv, pxchg = _px_kpi(latest, prev)

    kpi_items = [
        (f"{cat} Long",  _val(latest,lc,unit), _chg(latest,prev,lc,unit)),
        (f"{cat} Short", _val(latest,sc,unit), _chg(latest,prev,sc,unit)),
        (f"{cat} Net",   _val(latest,nc,unit), _chg(latest,prev,nc,unit)),
        ("Total OI", f"{latest.get('Total OI',np.nan)/1000:.1f}k"
                     if pd.notna(latest.get('Total OI')) else "—", ""),
        ("Price", pxv, pxchg),
    ]
    if spc and spc in d.columns:
        kpi_items.insert(3, (f"Spread", _val(latest,spc,unit), _chg(latest,prev,spc,unit)))
    kpi_row(kpi_items, color)

    ylabel = "k lots" if unit=="k lots" else "% of OI"
    r,g,b  = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    if view == "Net":
        yn = _get_y(d, nc, unit)
        traces = [{"trace": go.Scatter(
            x=d["Date"], y=yn, name=f"{cat} Net",
            fill="tozeroy",
            fillcolor=f"rgba({r},{g},{b},0.08)",
            line=dict(color=color, width=2.2),
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>Net: %{{y:.1f}}<extra></extra>",
        )}]
        ch1, ch2 = st.columns([3,1])
        with ch1: st.plotly_chart(timeseries(d,traces,f"{cat} Net  ·  {ylabel}",ylabel), use_container_width=True)
        with ch2:
            if nc in d.columns: st.plotly_chart(histogram_dist(d,nc,color,cat), use_container_width=True)
        st.plotly_chart(bars_weekly(d, nc, f"{cat} Net — weekly Δ  ·  k lots"), use_container_width=True)

    else:  # Legs
        traces = []
        if lc in d.columns:
            yl = _get_y(d,lc,unit)
            traces.append({"trace": go.Scatter(x=d["Date"],y=yl,name="Long",
                line=dict(color=C_LONG,width=2.0),
                hovertemplate="<b>%{x|%b %Y}</b><br>Long: %{y:.1f}<extra></extra>")})
        if sc in d.columns:
            ys = _get_y(d,sc,unit)
            traces.append({"trace": go.Scatter(x=d["Date"],y=ys,name="Short",
                line=dict(color=C_SHORT,width=2.0),
                hovertemplate="<b>%{x|%b %Y}</b><br>Short: %{y:.1f}<extra></extra>")})
        if spc and spc in d.columns:
            traces.append({"trace": go.Scatter(x=d["Date"],y=d[spc]/1000,name="Spread",
                line=dict(color="#94a3b8",width=1.4,dash="dot"),
                hovertemplate="<b>%{x|%b %Y}</b><br>Spread: %{y:.1f}k<extra></extra>")})
        st.plotly_chart(timeseries(d,traces,f"{cat} Legs  ·  {ylabel}",ylabel), use_container_width=True)
        ch1,ch2 = st.columns(2)
        with ch1:
            if lc in d.columns: st.plotly_chart(bars_weekly(d,lc,f"{cat} Long — weekly Δ"), use_container_width=True)
        with ch2:
            if sc in d.columns: st.plotly_chart(bars_weekly(d,sc,f"{cat} Short — weekly Δ"), use_container_width=True)

    with st.expander("Seasonality", expanded=False):
        scol = nc if view=="Net" else lc
        if scol in d.columns:
            st.plotly_chart(seasonal(d,scol,color,f"{cat} · {view}"), use_container_width=True)

    show_table(d, [lc,sc,nc,spc,"Total OI","Px"],
               [nc], f"Data table — {cat} ({ylabel})")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMMERCIAL
# ══════════════════════════════════════════════════════════════════════════════
def render_commercial(d, report, color):
    is_disagg = report == "Disagg"
    lc  = "Producer Long"  if is_disagg else "Comm Long"
    sc  = "Producer Short" if is_disagg else "Comm Short"
    nc  = "Comm Net"
    lbl = "Producer/Merchant" if is_disagg else "Commercial"

    unit = st.radio("Unit", ["k lots","% of OI"], horizontal=True, key="comm_unit")
    latest = d.iloc[-1]; prev = d.iloc[-2] if len(d)>1 else d.iloc[-1]
    pxv, pxchg = _px_kpi(latest, prev)

    kpi_items = [
        (f"{lbl} Long",  _val(latest,lc,unit), _chg(latest,prev,lc,unit)),
        (f"{lbl} Short", _val(latest,sc,unit), _chg(latest,prev,sc,unit)),
        (f"{lbl} Net",   _val(latest,nc,unit), _chg(latest,prev,nc,unit)),
        ("Total OI", f"{latest.get('Total OI',np.nan)/1000:.1f}k"
                     if pd.notna(latest.get('Total OI')) else "—", ""),
        ("Price", pxv, pxchg),
    ]
    kpi_row(kpi_items, color)

    ylabel = "k lots" if unit=="k lots" else "% of OI"
    r,g,b  = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    yl = _get_y(d,lc,unit); ys = _get_y(d,sc,unit); yn = _get_y(d,nc,unit)

    traces = [
        {"trace": go.Scatter(x=d["Date"],y=yl,name="Long",
            line=dict(color=C_LONG,width=2.0),
            hovertemplate="<b>%{x|%b %Y}</b><br>Long: %{y:.1f}<extra></extra>")},
        {"trace": go.Scatter(x=d["Date"],y=ys,name="Short",
            line=dict(color=C_SHORT,width=2.0),
            hovertemplate="<b>%{x|%b %Y}</b><br>Short: %{y:.1f}<extra></extra>")},
        {"trace": go.Scatter(x=d["Date"],y=yn,name="Net",
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.07)",
            line=dict(color=color,width=2.2),
            hovertemplate="<b>%{x|%b %Y}</b><br>Net: %{y:.1f}<extra></extra>")},
    ]
    ch1,ch2 = st.columns([3,1])
    with ch1: st.plotly_chart(timeseries(d,traces,f"{lbl} Long / Short  ·  {ylabel}",ylabel), use_container_width=True)
    with ch2: st.plotly_chart(histogram_dist(d,nc,color,lbl), use_container_width=True)

    ch1,ch2 = st.columns(2)
    with ch1: st.plotly_chart(bars_weekly(d,lc,f"{lbl} Long — weekly Δ"), use_container_width=True)
    with ch2: st.plotly_chart(bars_weekly(d,sc,f"{lbl} Short — weekly Δ"), use_container_width=True)

    with st.expander("Seasonality", expanded=False):
        ch1,ch2 = st.columns(2)
        with ch1: st.plotly_chart(seasonal(d,lc,C_LONG,f"{lbl} Long"), use_container_width=True)
        with ch2: st.plotly_chart(seasonal(d,sc,C_SHORT,f"{lbl} Short"), use_container_width=True)

    show_table(d, [lc,sc,nc,"Total OI","Px"], [nc], f"Data table — {lbl}")


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
    st.plotly_chart(timeseries(d,traces,f"Spreading by Category  ·  {ylabel}",ylabel), use_container_width=True)

    # Weekly change per category
    avail = [(lbl,col,clr) for lbl,(col,clr) in SPREAD_COLS.items() if col in d.columns]
    cols  = st.columns(len(avail))
    for i,(lbl,col,clr) in enumerate(avail):
        with cols[i]:
            st.plotly_chart(bars_weekly(d,col,f"{lbl} Spread — weekly Δ"), use_container_width=True)

    with st.expander("Seasonality", expanded=False):
        ch1,ch2,ch3 = st.columns(3)
        for ch,(lbl,col,clr) in zip([ch1,ch2,ch3], avail):
            with ch: st.plotly_chart(seasonal(d,col,clr,f"{lbl} Spread"), use_container_width=True)

    show_table(d, [col for _,(col,_) in SPREAD_COLS.items() if col in d.columns] + ["Total OI"],
               [col for _,(col,_) in SPREAD_COLS.items() if col in d.columns],
               "Data table — Spreading")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OLD / NEW CROP (Disagg only)
# ══════════════════════════════════════════════════════════════════════════════
def render_old_new(d_crops, color):
    old   = d_crops[d_crops["Crop"]=="Old"].sort_values("Date").reset_index(drop=True)
    other = d_crops[d_crops["Crop"]=="Other"].sort_values("Date").reset_index(drop=True)

    if old.empty and other.empty:
        st.info("No Old/Other crop data in selected range."); return

    # OI stacked bar
    dates    = old["Date"].combine_first(other["Date"]).sort_values().drop_duplicates()
    old_i    = old.set_index("Date");  other_i = other.set_index("Date")
    fig_oi = go.Figure([
        go.Bar(x=dates, y=old_i.reindex(dates)["Total OI"]/1000,
               name="Old Crop", marker=dict(color=C_OLD,opacity=0.85,line=dict(width=0)),
               hovertemplate="<b>%{x|%d %b %y}</b><br>Old OI: %{y:.1f}k<extra></extra>"),
        go.Bar(x=dates, y=other_i.reindex(dates)["Total OI"]/1000,
               name="New Crop", marker=dict(color=C_NEW,opacity=0.85,line=dict(width=0)),
               hovertemplate="<b>%{x|%d %b %y}</b><br>New OI: %{y:.1f}k<extra></extra>"),
    ])
    fig_oi.update_layout(**_BASE, barmode="stack", height=290,
        title=dict(text="Open Interest — Old vs New Crop  ·  k lots",font=dict(size=12,color="#333"),x=0),
        margin=dict(l=50,r=12,t=38,b=68),
        legend=dict(orientation="h",y=-0.24,x=0.5,xanchor="center",font_size=10),
        xaxis=dict(**_ax(x=True),tickformat="%d %b '%y"),
        yaxis=dict(**_ax(),title_text="k lots",title_font_size=10), bargap=0.18)
    st.plotly_chart(fig_oi, use_container_width=True)

    # Metric selector
    avail_metrics = [c for c in ["MM Net","Comm Net","MM Long","MM Short",
                                  "Producer Long","Producer Short",
                                  "Swap Net","Other Net","Non Rep Net"]
                     if c in old.columns or c in other.columns]
    ch1,ch2 = st.columns([2,5])
    with ch1: sel = st.selectbox("Metric", avail_metrics, key="on_metric")

    ch1,ch2 = st.columns(2)
    for widget, (crop_df, lbl, clr) in zip([ch1,ch2],[
        (old,   "Old Crop", C_OLD),
        (other, "New Crop", C_NEW),
    ]):
        with widget:
            if sel not in crop_df.columns: continue
            r,g,b = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=crop_df["Date"], y=crop_df[sel]/1000, name=lbl,
                fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.09)",
                line=dict(color=clr,width=2.2),
                hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>{lbl}: %{{y:.1f}}k<extra></extra>"))
            fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.12)")
            fig.update_layout(**_BASE, height=310,
                title=dict(text=f"{sel} — {lbl}  ·  k lots",font=dict(size=12,color="#333"),x=0),
                margin=dict(l=50,r=12,t=38,b=50), showlegend=False,
                xaxis=dict(**_ax(x=True),tickformat="%b '%y"),
                yaxis=dict(**_ax(),title_text="k lots",title_font_size=10))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Weekly Changes — Old vs New", expanded=False):
        ch1,ch2 = st.columns(2)
        with ch1:
            if sel in old.columns: st.plotly_chart(bars_weekly(old,sel,f"Old Crop {sel} — Δ"), use_container_width=True)
        with ch2:
            if sel in other.columns: st.plotly_chart(bars_weekly(other,sel,f"New Crop {sel} — Δ"), use_container_width=True)

    with st.expander("Seasonality — Old vs New", expanded=False):
        ch1,ch2 = st.columns(2)
        with ch1:
            if sel in old.columns: st.plotly_chart(seasonal(old,sel,C_OLD,f"Old · {sel}"), use_container_width=True)
        with ch2:
            if sel in other.columns: st.plotly_chart(seasonal(other,sel,C_NEW,f"New · {sel}"), use_container_width=True)

    tbl_cols = [c for c in ["MM Net","Comm Net","MM Long","MM Short",
                             "Producer Long","Producer Short","Total OI"] if c in old.columns]
    with st.expander("Data table — Old Crop", expanded=False):
        tbl = old[["Date"]+tbl_cols].copy()
        tbl["Date"] = pd.to_datetime(tbl["Date"]).dt.strftime("%d %b '%y")
        st.dataframe(tbl.sort_values("Date",ascending=False).reset_index(drop=True),
                     use_container_width=True, height=340)
    with st.expander("Data table — New Crop", expanded=False):
        tbl = other[["Date"]+tbl_cols].copy()
        tbl["Date"] = pd.to_datetime(tbl["Date"]).dt.strftime("%d %b '%y")
        st.dataframe(tbl.sort_values("Date",ascending=False).reset_index(drop=True),
                     use_container_width=True, height=340)


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
    st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fb, use_container_width=True)

    show_table(d, all_t, sel_cols, "Data table — trader counts")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CONCENTRATION
# ══════════════════════════════════════════════════════════════════════════════
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
    st.dataframe(pd.DataFrame(rows).set_index(""), use_container_width=False, height=130)
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
        st.plotly_chart(fig, use_container_width=True)

    show_table(d, avail, avail[:4], "Data table — Concentration ratios")


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
        st.dataframe(pd.DataFrame(rows).set_index("Position"), use_container_width=False)

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
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div style='background:#fff8e8;border:1px solid #fde68a;border-radius:8px;"
        "padding:10px 16px;margin-top:10px;font-size:.82rem;color:#92400e'>"
        "VaR module — reserved for integration from the existing VaR project.</div>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def render_analysis(d, report, color):
    net_opts = [c for c in
                (["Spec Net","Comm Net","Index Net","Non Rep Net","Combined Spec Net"]
                 if report=="CIT" else
                 ["MM Net","Comm Net","Swap Net","Other Net","Non Rep Net","Combined Spec Net"])
                if c in d.columns]

    st.markdown("#### Price vs Positioning")
    c1,_ = st.columns([2,5])
    with c1: sel = st.selectbox("COT element", net_opts, key="anal_col")

    if sel and "Px" in d.columns:
        ch1,ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(scatter_2d(d,"Px",sel,color,
                f"Price Δ%  vs  {sel} Δ","Price weekly Δ%",f"{sel} Δ (k lots)"),
                use_container_width=True)
        with ch2:
            # Level scatter
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
                    hovertemplate="<b>%{text}</b><br>Price: %{x:.2f}<br>Net: %{y:.1f}k<extra></extra>",
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
                    xaxis=dict(**_ax(x=True),title_text="Price"),
                    yaxis=dict(**_ax(),title_text=f"{sel} (k lots)"))
                st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Seasonality", expanded=False):
        c1,_ = st.columns([2,5])
        with c1: ssel = st.selectbox("Metric", net_opts, key="seas_col")
        if ssel in d.columns:
            st.plotly_chart(seasonal(d,ssel,color,ssel), use_container_width=True)

    with st.expander("COT vs COT Cross-Scatter", expanded=False):
        c1,c2 = st.columns(2)
        with c1: sx = st.selectbox("X axis", net_opts, index=0, key="xs_x")
        with c2: sy = st.selectbox("Y axis", net_opts,
                                   index=min(1,len(net_opts)-1), key="xs_y")
        if sx in d.columns and sy in d.columns:
            st.plotly_chart(scatter_2d(d,sx,sy,color,
                f"{sx} Δ  vs  {sy} Δ",f"{sx} Δ (k lots)",f"{sy} Δ (k lots)"),
                use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.05rem;font-weight:700;color:#1a56db;"
        "margin-bottom:16px;letter-spacing:.01em'>ICEBREAKER — COT</div>",
        unsafe_allow_html=True)

    commodity = st.selectbox("Commodity", list(COMM_NAMES.keys()),
                             format_func=lambda x: COMM_NAMES[x])
    color = COMM_COLORS[commodity]

    cit_ok = commodity in CIT_COMMS
    if cit_ok:
        report = st.radio("Report", ["CIT","Disagg"], horizontal=True)
    else:
        report = "Disagg"
        st.markdown("<div style='font-size:.73rem;color:#999;margin:-6px 0 8px'>"
                    "RC/LCC — Disaggregated only</div>", unsafe_allow_html=True)

    version_key = None
    if report == "Disagg":
        version = st.radio("Version", ["F&O combined","Fut only"], horizontal=True)
        version_key = "F&O" if "F&O" in version else "Fut"

    st.markdown("---")
    st.markdown("<div style='font-size:.78rem;font-weight:600;color:#444;"
                "margin-bottom:6px'>Date range</div>", unsafe_allow_html=True)
    start_date = st.date_input("From", value=datetime.date(2020,1,1), key="dt_from")
    end_date   = st.date_input("To",   value=datetime.date.today(),   key="dt_to")

    st.markdown("---")
    st.markdown("<div style='font-size:.68rem;color:#bbb;margin-top:4px'>"
                "Data: ICE Connect · CFTC<br>Updated Fridays</div>",
                unsafe_allow_html=True)


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
    raw = load_disagg(version_key)
    df_all_crops = raw[
        (raw["Commodity"]==commodity) &
        (raw["Date"]>=pd.Timestamp(start_date)) &
        (raw["Date"]<=pd.Timestamp(end_date))
    ].sort_values(["Crop","Date"]).reset_index(drop=True)
    df = df_all_crops[df_all_crops["Crop"]=="All"].sort_values("Date").reset_index(drop=True)


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
# TABS
# ══════════════════════════════════════════════════════════════════════════════
TAB_LABELS = ["Spec","Commercial","Spreading","Old / New",
              "Traders","Concentration","Exposure","Analysis"]

tabs = st.tabs(TAB_LABELS)

with tabs[0]: render_spec(df, report, color)
with tabs[1]: render_commercial(df, report, color)
with tabs[2]:
    if report=="Disagg": render_spreading(df, color)
    else: st.info("Spreading is only available for the Disaggregated report.")
with tabs[3]:
    if report=="Disagg" and df_all_crops is not None: render_old_new(df_all_crops, color)
    else: st.info("Old / New crop split is only available for the Disaggregated report.")
with tabs[4]: render_traders(df, report, color)
with tabs[5]: render_concentration(df, color)
with tabs[6]: render_exposure(df, commodity, color)
with tabs[7]: render_analysis(df, report, color)
