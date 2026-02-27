import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations
import io

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Return Decomposition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .stApp { background: #0d0f14; color: #e2e8f0; }
  section[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330; }
  section[data-testid="stSidebar"] * { color: #c8d0e0 !important; }
  h1 { font-family: 'IBM Plex Mono', monospace !important; color: #7dd3fc !important; font-size: 1.6rem !important; letter-spacing: -0.5px; margin-bottom: 2px !important; }
  h2 { font-family: 'IBM Plex Mono', monospace !important; color: #93c5fd !important; font-size: 1.3rem !important; }
  h3 { font-family: 'IBM Plex Mono', monospace !important; color: #93c5fd !important; font-size: 1.1rem !important; font-weight: 500 !important; }
  div[data-testid="metric-container"] { background: #151820; border: 1px solid #1e2a3a; border-radius: 8px; padding: 12px 16px; }
  div[data-testid="metric-container"] label { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #7dd3fc !important; font-family: 'IBM Plex Mono', monospace !important; }
  button[data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: #64748b !important; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #7dd3fc !important; border-bottom: 2px solid #7dd3fc !important; background: #151820 !important; }
  div[data-baseweb="select"] > div { background: #151820 !important; border: 1px solid #1e2a3a !important; color: #e2e8f0 !important; border-radius: 6px !important; }
  .streamlit-expanderHeader { color: #7dd3fc !important; font-family: 'IBM Plex Mono', monospace !important; }
  hr { border-color: #1e2a3a !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
PAPER_COLOR = "#111318"
BG_COLOR    = "#0d0f14"
GRID_COLOR  = "#1e2a3a"
PALETTE     = ["#7dd3fc","#93c5fd","#6ee7b7","#34d399","#fde68a","#fbbf24","#f9a8d4","#f472b6","#c4b5fd","#a78bfa"]
COMP_COLOR  = {"TR":"#7dd3fc","EPS":"#34d399","PE":"#f472b6","DY":"#fbbf24","Geometric":"#a78bfa","Arithmetic":"#fb923c"}
VAR_COLOR   = {"EPS":"#34d399","PE":"#f472b6","DY":"#fbbf24"}
ACCENT      = ["#7dd3fc","#34d399","#f472b6","#fbbf24","#a78bfa","#fb923c"]
ALL_H       = [f"{y}Y" for y in range(1, 11)]
COMP_LABEL  = {"TR":"Total Return","EPS":"EPS Growth","PE":"P/E Re-rating","DY":"Dividend Yield"}

# Expected fixed column names in every sheet
REQUIRED_COLS = ["date", "tri", "pe", "div_yield_pct", "eps"]

def base_layout(title="", height=460, yaxis_fmt=".0%"):
    """Returns a mutable dict — callers modify keys before passing to update_layout()."""
    return dict(
        template="plotly_dark",
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        height=height,
        title=dict(text=title, font=dict(family="IBM Plex Mono", size=14, color="#93c5fd"), x=0.02),
        font=dict(family="IBM Plex Sans", color="#c8d0e0"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=GRID_COLOR, borderwidth=1,
                    font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=55, r=20, t=65, b=45),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showspikes=True,
                   spikecolor="#374151", spikethickness=1),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                   tickformat=yaxis_fmt, showspikes=True),
    )

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Config")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

    # Sheet / index selector — populated after file is uploaded
    selected_sheet = None
    if uploaded_file is not None:
        try:
            xf = pd.ExcelFile(io.BytesIO(uploaded_file.read()))
            available_sheets = xf.sheet_names
            uploaded_file.seek(0)          # reset so pd.read_excel still works later
        except Exception as e:
            st.error(f"Could not read sheet names: {e}")
            available_sheets = []

        if available_sheets:
            st.markdown("---")
            st.markdown("**Select Index**")
            selected_sheet = st.selectbox(
                "Index (sheet)",
                available_sheets,
                help="Each sheet represents one index. Columns must be: date, tri, pe, div_yield_pct, eps",
            )

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 📊 Return Decomposition Dashboard")
st.markdown(
    "<p style='color:#64748b;font-size:13px;margin-top:-10px;'>"
    "TR = EPS Growth + P/E Re-rating + Dividend Yield  ·  Rolling 1Y–10Y  ·  "
    "Columns expected: <code style='color:#7dd3fc'>date · tri · pe · div_yield_pct · eps</code></p>",
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("👈 Upload an Excel file in the sidebar to begin.")
    st.stop()

if not selected_sheet:
    st.warning("No sheets found in the uploaded file.")
    st.stop()

# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Processing data…")
def load_and_process(file_bytes, sheet):
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)

    # Validate required columns exist (case-insensitive)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Sheet '{sheet}' is missing columns: {missing}. "
            f"Found: {list(raw.columns)}"
        )

    df = raw[REQUIRED_COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = (df.sort_values("date")
            .dropna(subset=["pe", "eps", "tri", "div_yield_pct"])
            .reset_index(drop=True)
            .set_index("date"))

    df["TR_m"]  = df["tri"].pct_change()
    df["EPS_m"] = df["eps"].pct_change()
    df["PE_m"]  = df["pe"].pct_change()
    df["DY_m"]  = df["div_yield_pct"] / 100 / 12

    growth_dict = {}
    for y in range(1, 11):
        w = 12 * y
        dh = pd.DataFrame(index=df.index)
        dh["TR"]  = (1 + df["TR_m"]).rolling(w).apply(np.prod, raw=True) ** (1/y) - 1
        dh["EPS"] = (1 + df["EPS_m"]).rolling(w).apply(np.prod, raw=True) ** (1/y) - 1
        dh["PE"]  = (1 + df["PE_m"]).rolling(w).apply(np.prod, raw=True) ** (1/y) - 1
        dh["DY"]  = (1 + df["DY_m"]).rolling(w).apply(np.prod, raw=True) ** (1/y) - 1
        dh["Geometric"] = (
            (1 + df["EPS_m"]) * (1 + df["PE_m"]) * (1 + df["DY_m"])
        ).rolling(w).apply(np.prod, raw=True) ** (1/y) - 1
        dh["Arithmetic"] = (
            df["EPS_m"].rolling(w).mean() * 12 +
            df["PE_m"].rolling(w).mean()  * 12 +
            df["DY_m"].rolling(w).mean()  * 12
        )
        growth_dict[f"{y}Y"] = dh

    corr_dict = {h: gd[["TR","EPS","PE","DY"]].dropna().corr() for h, gd in growth_dict.items()}
    return df, growth_dict, corr_dict

try:
    file_bytes = uploaded_file.read()
    df, growth_dict, corr_dict = load_and_process(file_bytes, selected_sheet)
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# ─── Metrics row ──────────────────────────────────────────────────────────────
st.markdown(
    f"<p style='color:#7dd3fc;font-family:IBM Plex Mono,monospace;font-size:15px;"
    f"margin-bottom:8px;'>📌 {selected_sheet}</p>",
    unsafe_allow_html=True,
)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Start Date",   str(df.index.min().date()))
c2.metric("End Date",     str(df.index.max().date()))
c3.metric("Observations", f"{len(df):,}")
c4.metric("Latest P/E",   f"{df['pe'].iloc[-1]:.1f}×")
c5.metric("Latest DY",    f"{df['div_yield_pct'].iloc[-1]:.2f}%")
st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Attribution",
    "🌡️  Correlation Matrix",
    "📉  Corr vs Horizon",
    "🔭  Fan Charts",
    "🧩  Variance Decomposition",
    "📊  Distribution & Stats",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Attribution
# ══════════════════════════════════════════════════════════════════
with tab1:
    cc, _ = st.columns([1, 3])
    sel_h = cc.selectbox("Horizon", ALL_H, index=4, key="attr_h")
    df_h  = growth_dict[sel_h].dropna()

    fig = go.Figure()
    for col, lbl, w, dash in [
        ("TR",         "Total Return",        3,   None),
        ("EPS",        "EPS Growth",          1.8, None),
        ("PE",         "P/E Re-rating",       1.8, None),
        ("DY",         "Dividend Yield",      1.8, None),
        ("Arithmetic", "Arithmetic Sum",      2,   "dash"),
        ("Geometric",  "Geometric Combined",  2,   "dot"),
    ]:
        lkw = dict(color=COMP_COLOR[col], width=w)
        if dash: lkw["dash"] = dash
        fig.add_trace(go.Scatter(
            x=df_h.index, y=df_h[col], name=lbl, line=lkw,
            hovertemplate=f"<b>{lbl}</b>  %{{x|%b %Y}} → %{{y:.2%}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="#374151", line_width=1)
    fig.update_layout(**base_layout(f"{sel_h} Rolling Annualised Attribution — {selected_sheet}", 500))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Latest values"):
        snap = df_h.tail(1).T.rename(columns={df_h.index[-1]: "Latest CAGR"})
        snap["Latest CAGR"] = snap["Latest CAGR"].map("{:.2%}".format)
        st.dataframe(snap, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Correlation Matrix
# ══════════════════════════════════════════════════════════════════
with tab2:
    cc2, _ = st.columns([1, 3])
    sel_ch = cc2.selectbox("Horizon", ALL_H, index=4, key="corr_h")
    corr   = corr_dict[sel_ch]
    labels_c = list(corr.columns)
    z    = corr.values
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig2 = go.Figure(go.Heatmap(
        z=z, x=labels_c, y=labels_c,
        text=text, texttemplate="%{text}",
        colorscale=[[0,"#f472b6"],[0.5,"#1e2a3a"],[1,"#34d399"]],
        zmid=0, zmin=-1, zmax=1,
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(tickformat=".1f", thickness=14, len=0.8),
    ))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
        height=440,
        title=dict(text=f"{sel_ch} Correlation Matrix — {selected_sheet}",
                   font=dict(family="IBM Plex Mono", size=14, color="#93c5fd"), x=0.02),
        font=dict(family="IBM Plex Sans", color="#c8d0e0"),
        margin=dict(l=50, r=20, t=65, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 Raw table"):
        st.dataframe(
            corr.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Corr vs Horizon
# ══════════════════════════════════════════════════════════════════
with tab3:
    pairs = list(combinations(["TR","EPS","PE","DY"], 2))
    rows  = []
    for h, cm in corr_dict.items():
        yr  = int(h.replace("Y",""))
        row = {"Years": yr}
        for a, b in pairs:
            row[f"{a}-{b}"] = cm.loc[a, b]
        rows.append(row)
    ch_df = pd.DataFrame(rows).sort_values("Years").set_index("Years")

    cc3, _ = st.columns([1, 3])
    sel_pairs = cc3.multiselect(
        "Pairs", ch_df.columns.tolist(),
        default=["TR-EPS","TR-PE","EPS-PE"], key="pairs_ms",
    )

    fig3 = go.Figure()
    for i, col in enumerate(sel_pairs):
        fig3.add_trace(go.Scatter(
            x=ch_df.index, y=ch_df[col], name=col,
            mode="lines+markers",
            line=dict(color=ACCENT[i % len(ACCENT)], width=2),
            marker=dict(size=7),
            hovertemplate=f"<b>{col}</b><br>Horizon: %{{x}}Y<br>r = %{{y:.3f}}<extra></extra>",
        ))
    fig3.add_hline(y=0, line_color="#374151", line_width=1)
    lo3 = base_layout(f"Pairwise Correlation vs Horizon — {selected_sheet}", 450)
    lo3["xaxis"]["title"] = "Horizon (Years)"
    lo3["xaxis"]["dtick"] = 1
    lo3["yaxis"]["title"] = "Correlation"
    lo3["yaxis"]["range"] = [-1, 1]
    lo3["yaxis"]["tickformat"] = ".2f"
    fig3.update_layout(**lo3)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — Fan Charts
# ══════════════════════════════════════════════════════════════════
with tab4:
    cc4, _ = st.columns([1, 3])
    fan_p  = cc4.selectbox("Component", ["TR","EPS","PE","DY"],
                            format_func=lambda x: COMP_LABEL[x], key="fan_p")
    fig4 = go.Figure()
    for y in range(1, 11):
        df_y = growth_dict[f"{y}Y"]
        fig4.add_trace(go.Scatter(
            x=df_y.index, y=df_y[fan_p], name=f"{y}Y",
            line=dict(color=PALETTE[y-1], width=1.5),
            hovertemplate=f"<b>{y}Y</b>  %{{x|%b %Y}} → %{{y:.2%}}<extra></extra>",
        ))
    fig4.add_hline(y=0, line_color="#374151", line_width=1)
    fig4.update_layout(**base_layout(
        f"{COMP_LABEL[fan_p]} — Rolling CAGR Fan 1Y–10Y — {selected_sheet}", 500))
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — Variance Decomposition
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Static Variance Decomposition by Horizon")
    st.markdown(
        "<p style='color:#64748b;font-size:12px;'>Share of total component variance (EPS + PE + DY) "
        "attributable to each driver across the full sample. PE dominates short horizons; "
        "EPS takes over at longer ones.</p>", unsafe_allow_html=True,
    )

    var_rows = []
    for y in range(1, 11):
        temp = growth_dict[f"{y}Y"].dropna()
        ve   = temp["EPS"].var()
        vp   = temp["PE"].var()
        vd   = temp["DY"].var()
        tot  = ve + vp + vd
        var_rows.append({"Years": y, "EPS": ve/tot, "PE": vp/tot, "DY": vd/tot})
    var_df = pd.DataFrame(var_rows).set_index("Years")

    fig_var = go.Figure()
    for comp in ["EPS", "PE", "DY"]:
        fig_var.add_trace(go.Bar(
            x=ALL_H, y=var_df[comp],
            name=COMP_LABEL[comp],
            marker_color=VAR_COLOR[comp],
            hovertemplate=f"<b>{COMP_LABEL[comp]}</b><br>Horizon: %{{x}}<br>Share: %{{y:.1%}}<extra></extra>",
        ))
    lo_var = base_layout(f"Static Variance Decomposition — {selected_sheet}", 460)
    lo_var["barmode"]         = "stack"
    lo_var["xaxis"]["title"]  = "Horizon"
    lo_var["yaxis"]["range"]  = [0, 1]
    fig_var.update_layout(**lo_var)
    st.plotly_chart(fig_var, use_container_width=True)

    with st.expander("📋 Variance share table"):
        disp         = (var_df * 100).round(1).copy()
        disp.index   = ALL_H
        disp.columns = ["EPS (%)", "PE (%)", "DY (%)"]
        st.dataframe(
            disp.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=1),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Time-Varying (Rolling) Variance Decomposition")
    st.markdown(
        "<p style='color:#64748b;font-size:12px;'>Variance shares recomputed over a rolling window equal to "
        "the selected horizon — revealing regime shifts where PE or EPS dominated.</p>",
        unsafe_allow_html=True,
    )

    rv_cc, _ = st.columns([1, 3])
    rv_h     = rv_cc.selectbox("Horizon", ALL_H, index=4, key="rv_h")
    y_rv     = int(rv_h.replace("Y",""))
    window_rv = y_rv * 12

    df_rv = growth_dict[rv_h][["EPS","PE","DY"]].dropna()
    ve_r  = df_rv["EPS"].rolling(window_rv).var()
    vp_r  = df_rv["PE"].rolling(window_rv).var()
    vd_r  = df_rv["DY"].rolling(window_rv).var()
    tot_r = ve_r + vp_r + vd_r

    fig_rv = go.Figure()
    for comp, series in [("EPS", ve_r/tot_r), ("PE", vp_r/tot_r), ("DY", vd_r/tot_r)]:
        fig_rv.add_trace(go.Scatter(
            x=series.index, y=series,
            name=COMP_LABEL[comp],
            line=dict(color=VAR_COLOR[comp], width=2),
            hovertemplate=f"<b>{COMP_LABEL[comp]}</b>  %{{x|%b %Y}} → %{{y:.1%}}<extra></extra>",
        ))
    fig_rv.add_hline(y=0, line_color="#374151", line_width=1)
    lo_rv = base_layout(f"{rv_h} Rolling Variance Contribution — {selected_sheet}", 430)
    lo_rv["xaxis"]["title"] = "Date"
    lo_rv["yaxis"]["title"] = "Variance Share"
    fig_rv.update_layout(**lo_rv)
    st.plotly_chart(fig_rv, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 6 — Distribution & Stats
# ══════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Rolling CAGR Distributions")

    dc1, dc2  = st.columns([1, 1])
    dist_comp = dc1.selectbox("Component", ["TR","EPS","PE","DY"],
                               format_func=lambda x: COMP_LABEL[x], key="dist_comp")
    dist_type = dc2.selectbox("Chart Type", ["Violin","Box","Histogram + KDE"], key="dist_type")
    st.markdown("---")

    if dist_type == "Violin":
        fig5 = go.Figure()
        for y in range(1, 11):
            vals = growth_dict[f"{y}Y"][dist_comp].dropna()
            fig5.add_trace(go.Violin(
                y=vals, name=f"{y}Y",
                box_visible=True, meanline_visible=True,
                fillcolor=PALETTE[y-1], opacity=0.75,
                line_color="white", line_width=0.8,
                points="outliers",
                hovertemplate="<b>%{fullData.name}</b><br>%{y:.2%}<extra></extra>",
            ))
        lo5 = base_layout(f"{COMP_LABEL[dist_comp]} — Violin Distributions (1Y–10Y)", 540)
        lo5["violingap"]        = 0.12
        lo5["showlegend"]       = False
        lo5["xaxis"]["title"]   = "Horizon"
        fig5.update_layout(**lo5)
        st.plotly_chart(fig5, use_container_width=True)

    elif dist_type == "Box":
        fig5 = go.Figure()
        for y in range(1, 11):
            vals = growth_dict[f"{y}Y"][dist_comp].dropna()
            fig5.add_trace(go.Box(
                y=vals, name=f"{y}Y",
                marker_color=PALETTE[y-1],
                line_color="white", line_width=0.8,
                boxmean="sd",
                hovertemplate="<b>%{fullData.name}</b><br>%{y:.2%}<extra></extra>",
            ))
        lo5 = base_layout(f"{COMP_LABEL[dist_comp]} — Box Plots (1Y–10Y)", 520)
        lo5["showlegend"]     = False
        lo5["xaxis"]["title"] = "Horizon"
        fig5.update_layout(**lo5)
        st.plotly_chart(fig5, use_container_width=True)

    else:  # Histogram + KDE
        hist_h   = st.select_slider("Horizon", ALL_H, value="5Y", key="hist_h")
        vals_pct = growth_dict[hist_h][dist_comp].dropna() * 100

        from scipy.stats import gaussian_kde
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(
            x=vals_pct, name="Frequency",
            nbinsx=45, marker_color=COMP_COLOR[dist_comp], opacity=0.5,
            histnorm="probability density",
            hovertemplate="CAGR: %{x:.1f}%<br>Density: %{y:.4f}<extra></extra>",
        ))
        kde = gaussian_kde(vals_pct)
        xr  = np.linspace(vals_pct.min(), vals_pct.max(), 400)
        fig5.add_trace(go.Scatter(
            x=xr, y=kde(xr), name="KDE",
            line=dict(color="#f8fafc", width=2.5),
        ))
        med = vals_pct.median()
        fig5.add_vline(x=med, line_color="#fbbf24", line_width=1.5, line_dash="dash",
                       annotation_text=f"Median {med:.1f}%",
                       annotation_font_color="#fbbf24", annotation_position="top right")
        fig5.add_vline(x=0, line_color="#f472b6", line_width=1, line_dash="dot")
        lo5 = base_layout(
            f"{COMP_LABEL[dist_comp]} — {hist_h} CAGR Distribution", 440, yaxis_fmt=".4f")
        lo5["xaxis"]["title"] = "Rolling CAGR (%)"
        lo5["yaxis"]["title"] = "Density"
        lo5["hovermode"]      = "x"
        fig5.update_layout(**lo5)
        st.plotly_chart(fig5, use_container_width=True)

    # Descriptive Stats table
    st.markdown("### 📋 Descriptive Statistics — All Horizons")
    stats_rows = []
    for y in range(1, 11):
        s = growth_dict[f"{y}Y"][dist_comp].dropna()
        stats_rows.append({
            "Horizon":    f"{y}Y",
            "Mean":       f"{s.mean():.2%}",
            "Median":     f"{s.median():.2%}",
            "Std Dev":    f"{s.std():.2%}",
            "Min":        f"{s.min():.2%}",
            "Max":        f"{s.max():.2%}",
            "5th Pctl":   f"{s.quantile(0.05):.2%}",
            "95th Pctl":  f"{s.quantile(0.95):.2%}",
            "% Positive": f"{(s > 0).mean():.1%}",
            "Skewness":   f"{s.skew():.2f}",
            "Kurtosis":   f"{s.kurt():.2f}",
            "N":          len(s),
        })
    st.dataframe(pd.DataFrame(stats_rows).set_index("Horizon"), use_container_width=True)

    # Mean ± Std bar
    st.markdown("### 📊 Mean CAGR ± 1 Std Dev by Horizon")
    means   = [growth_dict[f"{y}Y"][dist_comp].dropna().mean() for y in range(1, 11)]
    stds    = [growth_dict[f"{y}Y"][dist_comp].dropna().std()  for y in range(1, 11)]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=ALL_H, y=means,
        error_y=dict(type="data", array=stds, visible=True, color="#64748b", thickness=1.5),
        marker_color=[COMP_COLOR[dist_comp]] * 10,
        marker_line_color="white", marker_line_width=0.5,
        hovertemplate="<b>%{x}</b><br>Mean: %{y:.2%}<extra></extra>",
    ))
    fig_bar.add_hline(y=0, line_color="#374151", line_width=1)
    lo_bar = base_layout(f"{COMP_LABEL[dist_comp]} — Mean CAGR ± 1σ by Horizon", 380)
    lo_bar["showlegend"]     = False
    lo_bar["xaxis"]["title"] = "Horizon"
    fig_bar.update_layout(**lo_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

    # % Positive heatmap
    st.markdown("### 🎯 % Positive Periods — All Components × All Horizons")
    comps = ["TR","EPS","PE","DY"]
    hz    = [
        [round((growth_dict[f"{y}Y"][c].dropna() > 0).mean() * 100, 1) for y in range(1, 11)]
        for c in comps
    ]
    fig_hm = go.Figure(go.Heatmap(
        z=hz, x=ALL_H, y=[COMP_LABEL[c] for c in comps],
        text=[[f"{v:.0f}%" for v in row] for row in hz],
        texttemplate="%{text}",
        colorscale=[[0,"#f472b6"],[0.5,"#1e2a3a"],[1,"#34d399"]],
        zmid=50, zmin=0, zmax=100,
        hovertemplate="<b>%{y}</b> · <b>%{x}</b><br>Positive: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="% Positive", ticksuffix="%", thickness=14),
    ))
    fig_hm.update_layout(
        template="plotly_dark", paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
        height=280,
        title=dict(text="% of Rolling Periods with Positive CAGR",
                   font=dict(family="IBM Plex Mono", size=14, color="#93c5fd"), x=0.02),
        font=dict(family="IBM Plex Sans", color="#c8d0e0"),
        margin=dict(l=130, r=20, t=65, b=40),
        xaxis_title="Horizon",
    )
    st.plotly_chart(fig_hm, use_container_width=True)
