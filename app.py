import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FEMA Disaster Intelligence",
    page_icon="📊",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — deep navy + ember accent palette
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:    #020916;
    --bg-card:    #071428;
    --bg-card2:   #0b1e3d;
    --border:     #122a52;
    --accent1:    #ff6b35;
    --accent2:    #f7c59f;
    --accent3:    #00c9ff;
    --accent4:    #ff3860;
    --text-hi:    #e8f4fd;
    --text-mid:   #8baabf;
    --text-lo:    #3d5a73;
    --glow:       rgba(255,107,53,0.15);
}

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif !important;
    background: var(--bg-deep) !important;
    color: var(--text-hi) !important;
}

.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,120,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(255,107,53,0.06) 0%, transparent 55%),
        var(--bg-deep) !important;
    min-height: 100vh;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-hi) !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--accent1) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
}

/* ── inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-hi) !important;
    border-radius: 8px !important;
}
.stSlider .stSlider { color: var(--accent1) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent1) !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px 12px 0 0 !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 4px !important;
    padding: 4px 8px 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-mid) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 18px !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent1) !important;
    background: rgba(255,107,53,0.1) !important;
    border-bottom: 2px solid var(--accent1) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-card) !important;
    border-radius: 0 0 12px 12px !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    padding: 24px !important;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 16px; margin-bottom: 24px; }
.kpi-card {
    flex: 1;
    background: linear-gradient(135deg, var(--bg-card2) 0%, #0d2045 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 18px 16px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px var(--glow);
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-c1::after { background: linear-gradient(90deg, #ff6b35, #ff9f68); }
.kpi-c2::after { background: linear-gradient(90deg, #00c9ff, #0099cc); }
.kpi-c3::after { background: linear-gradient(90deg, #ff3860, #ff6b8a); }
.kpi-c4::after { background: linear-gradient(90deg, #7c5cbf, #a07de8); }
.kpi-c5::after { background: linear-gradient(90deg, #00bf72, #00e68a); }
.kpi-c6::after { background: linear-gradient(90deg, #f7c519, #ffd966); }

.kpi-icon { font-size: 1.5rem; margin-bottom: 8px; }
.kpi-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-hi);
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-label {
    font-size: 0.68rem;
    color: var(--text-mid);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 500;
}
.kpi-sub {
    font-size: 0.72rem;
    color: var(--accent2);
    margin-top: 4px;
    opacity: 0.8;
}

/* ── section headers ── */
.sec-head {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    color: var(--accent1);
    text-transform: uppercase;
    padding: 6px 0 6px 14px;
    border-left: 3px solid var(--accent1);
    margin: 20px 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── chart wrapper ── */
.chart-wrap {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 4px;
}

/* ── hero header ── */
.hero {
    text-align: center;
    padding: 28px 0 16px;
    position: relative;
}
.hero h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.12em !important;
    background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 40%, #00c9ff 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 !important;
    line-height: 1.1 !important;
}
.hero-sub {
    color: var(--text-mid);
    font-size: 0.82rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 8px;
}
.hero-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent1), var(--accent3), transparent);
    margin: 16px auto 0;
    width: 60%;
    border-radius: 2px;
}

/* ── divider ── */
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent1); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY BASE THEME
# ══════════════════════════════════════════════════════════════════════════════
BG        = "rgba(0,0,0,0)"
GRID      = "#122a52"
TEXT_HI   = "#e8f4fd"
TEXT_MID  = "#8baabf"
TEXT_LO   = "#3d5a73"

BASE_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="Exo 2", color=TEXT_MID, size=12),
    title=dict(font=dict(family="Orbitron", color=TEXT_HI, size=13), x=0.02, xanchor="left"),
    legend=dict(
        bgcolor="rgba(7,20,40,0.9)", bordercolor=GRID, borderwidth=1,
        font=dict(color=TEXT_HI, size=11), title=dict(font=dict(color=TEXT_HI)),
    ),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
               tickfont=dict(color=TEXT_LO), title=dict(font=dict(color=TEXT_MID))),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
               tickfont=dict(color=TEXT_LO), title=dict(font=dict(color=TEXT_MID))),
    margin=dict(l=45, r=20, t=50, b=45),
    hoverlabel=dict(bgcolor="#0b1e3d", bordercolor=GRID, font=dict(color=TEXT_HI, family="Exo 2")),
)

PALETTE = ["#ff6b35","#00c9ff","#ff3860","#7c5cbf","#00bf72",
           "#f7c519","#ec4899","#38bdf8","#84cc16","#fb923c",
           "#a78bfa","#34d399","#f472b6","#60a5fa","#facc15"]

def apply(fig, h=380, **kwargs):
    fig.update_layout(**BASE_LAYOUT, height=h, **kwargs)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# DATA  
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load():
    df = pd.read_csv("data.csv", low_memory=False)
    for c in ["declarationDate","incidentBeginDate","incidentEndDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["declarationDate"])
    df["year"]       = df["declarationDate"].dt.year.astype(int)
    df["month"]      = df["declarationDate"].dt.month.astype(int)
    df["month_name"] = df["declarationDate"].dt.strftime("%b")
    df["quarter"]    = df["declarationDate"].dt.quarter
    df["decade"]     = (df["year"] // 10 * 10).astype(str) + "s"
    df["incidentType"] = df["incidentType"].str.strip().str.title().fillna("Unknown")
    df["state"]        = df["state"].str.strip().str.upper()
    df["declarationType"] = df["declarationType"].str.strip().str.upper()

    valid = {"AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
             "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
             "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
             "TX","UT","VT","VA","WA","WV","WI","WY","DC"}
    df = df[df["state"].isin(valid)].copy()

    region_map = {
        "AL":"Southeast","GA":"Southeast","MS":"Southeast","SC":"Southeast","NC":"Southeast","TN":"Southeast",
        "TX":"Gulf Coast","FL":"Gulf Coast","LA":"Gulf Coast",
        "CA":"West Coast","OR":"West Coast","WA":"West Coast",
        "AK":"Pacific Northwest","HI":"Pacific Northwest",
        "CO":"Mountain","UT":"Mountain","AZ":"Mountain","NM":"Mountain","NV":"Mountain","MT":"Mountain","ID":"Mountain","WY":"Mountain",
        "NY":"Northeast","PA":"Northeast","MA":"Northeast","CT":"Northeast","NJ":"Northeast","ME":"Northeast","NH":"Northeast","VT":"Northeast","RI":"Northeast",
        "OH":"Great Lakes","MI":"Great Lakes","IN":"Great Lakes","IL":"Great Lakes","WI":"Great Lakes","MN":"Great Lakes",
        "OK":"Midwest","MO":"Midwest","KS":"Midwest","NE":"Midwest","IA":"Midwest","ND":"Midwest","SD":"Midwest",
        "VA":"Mid-Atlantic","MD":"Mid-Atlantic","DE":"Mid-Atlantic","WV":"Mid-Atlantic","DC":"Mid-Atlantic",
        "KY":"Appalachia","AR":"Appalachia",
    }
    df["region_grp"] = df["state"].map(region_map).fillna("Other")
    return df

df = load()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # st.markdown("### 🌪️ FEMA DASHBOARD")
    st.markdown("### 📊 FEMA DASHBOARD")
    st.markdown("---")

    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    yr = st.slider("📅 Year Range", yr_min, yr_max, (yr_min, yr_max))

    all_types = sorted(df["incidentType"].unique())
    top10     = df["incidentType"].value_counts().head(10).index.tolist()
    sel_types = st.multiselect("🔥 Incident Types", all_types, default=top10,
                               help="Empty = all types")

    all_states  = sorted(df["state"].unique())
    sel_states  = st.multiselect("🗺️ Filter States", all_states, default=[],
                                 help="Empty = all states")

    dec_types   = st.multiselect("📋 Declaration Type",
                                  sorted(df["declarationType"].unique()),
                                  default=sorted(df["declarationType"].unique()))

    top_n = st.slider("🏆 Top N States", 5, 30, 15)

    st.markdown("---")
    st.markdown(
        "<small style='color:#3d5a73'>FEMA Disaster Declarations<br>"
        "68,000+ records · 1953–2025<br>Built with Streamlit + Plotly</small>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# FILTER
# ══════════════════════════════════════════════════════════════════════════════
flt = df[df["year"].between(*yr)]
if sel_types:   flt = flt[flt["incidentType"].isin(sel_types)]
if sel_states:  flt = flt[flt["state"].isin(sel_states)]
if dec_types:   flt = flt[flt["declarationType"].isin(dec_types)]
# ── RISK SCORING ──
recent = flt[flt["year"] >= flt["year"].max() - 10]

total_counts = flt.groupby("state").size()
recent_counts = recent.groupby("state").size()

risk_df = pd.DataFrame({
    "total": total_counts,
    "recent": recent_counts
}).fillna(0)

risk_df["risk_score"] = (risk_df["total"] * 0.5) + (risk_df["recent"] * 0.5)
risk_df = risk_df.sort_values("risk_score", ascending=False)
risk_df["risk_level"] = pd.qcut(
    risk_df["risk_score"],
    q=3,
    labels=["Low", "Medium", "High"]
)
# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>FEMA Disaster Intelligence Platform: Risk Analysis & Forecasting</h1>
  <div class="hero-sub">A Decision-Support Dashboard for Analyzing, Predicting, and Managing Disaster Risks</div>
  <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════
# GLOBAL CALCULATIONS (ADD HERE)
# ══════════════════════════════════════════════════════════════════════

yearly = flt.groupby("year").size().reset_index(name="count")
yearly["yoy"] = yearly["count"].pct_change() * 100

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
total      = len(flt)
top_state  = flt["state"].value_counts().idxmax()  if total else "N/A"
top_inc    = flt["incidentType"].value_counts().idxmax() if total else "N/A"
peak_yr    = int(flt.groupby("year").size().idxmax()) if total else "N/A"
pct_pa     = flt["paProgramDeclared"].mean()*100 if "paProgramDeclared" in flt else 0
unique_inc = flt["incidentType"].nunique()

cols = st.columns(6)
kpis = [
    ("🌊", f"{total:,}", "Total Declarations", f"{yr[0]}–{yr[1]}", "kpi-c1"),
    ("🗓️", str(flt["year"].nunique()), "Years Covered", "in selected range", "kpi-c2"),
    ("📍", top_state, "Most Affected State", "highest count", "kpi-c3"),
    ("⚡", top_inc, "Top Incident", "most frequent type", "kpi-c4"),
    ("📈", str(peak_yr), "Peak Year", "most declarations", "kpi-c5"),
    ("🏷️", str(unique_inc), "Incident Types", "distinct categories", "kpi-c6"),
]
for col, (icon, val, label, sub, cls) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔍 Key Insights")

avg_growth = yearly["yoy"].mean() if "yoy" in yearly else 0

st.write(f"""
- {top_state} is the most affected state  
- {top_inc} is the most frequent disaster  
- Peak year: {peak_yr}  
- Avg growth: {avg_growth:.2f}%  
""")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📈 Temporal Trends",
    "🗺️ Geographic Patterns",
    "🔬 Incident Analysis",
    "📊 Statistical Insights",
    "🤝 Assistance Programs",
    "🎯 Risk Insights"
])
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TEMPORAL TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with t1:
    
    # ── FORECASTING ──
    if len(yearly) > 5:
        X = yearly["year"].values.reshape(-1, 1)
        y = yearly["count"].values

        model = LinearRegression()
        model.fit(X, y)

        future_years = np.arange(yearly["year"].max()+1, yearly["year"].max()+6).reshape(-1,1)
        predictions = model.predict(future_years)

        future_df = pd.DataFrame({
            "year": future_years.flatten(),
            "predicted": predictions
    })
    # ── FORECASTING ──

    yearly["rolling5"] = yearly["count"].rolling(5, center=True, min_periods=1).mean()
    yearly["rolling10"] = yearly["count"].rolling(10, center=True, min_periods=1).mean()
    # yearly["yoy"] = yearly["count"].pct_change() * 100
    yearly["cumsum"] = yearly["count"].cumsum()
    yearly["decade"] = (yearly["year"] // 10 * 10).astype(str) + "s"

    st.markdown('<div class="sec-head">01 — ANNUAL DECLARATIONS TIMELINE</div>', unsafe_allow_html=True)

    # 1. Area + line combo
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["count"],
        fill="tozeroy", name="Declarations",
        line=dict(color="#ff6b35", width=1.5),
        fillcolor="rgba(255,107,53,0.12)",
        hovertemplate="<b>%{x}</b><br>Declarations: %{y:,}<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["rolling5"],
        name="5-yr Avg", line=dict(color="#00c9ff", width=2.5, dash="dot"),
        hovertemplate="5yr avg: %{y:.0f}<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["rolling10"],
        name="10-yr Avg", line=dict(color="#f7c519", width=2, dash="dash"),
        hovertemplate="10yr avg: %{y:.0f}<extra></extra>",
    ))
    apply(fig1, h=360, title_text="Annual FEMA Disaster Declarations with Rolling Averages")
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-head">02 — YEAR-OVER-YEAR GROWTH RATE</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        colors_yoy = ["#ff3860" if v < 0 else "#00bf72" for v in yearly["yoy"].fillna(0)]
        fig2.add_trace(go.Bar(
            x=yearly["year"], y=yearly["yoy"],
            marker_color=colors_yoy,
            hovertemplate="<b>%{x}</b><br>YoY: %{y:.1f}%<extra></extra>",
            name="Growth %",
        ))
        fig2.add_hline(y=0, line_color="#8baabf", line_width=1)
        apply(fig2, h=300, title_text="Year-over-Year Growth Rate (%)", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-head">03 — CUMULATIVE DECLARATIONS</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["cumsum"],
            fill="tozeroy", fillcolor="rgba(0,201,255,0.08)",
            line=dict(color="#00c9ff", width=2.5),
            hovertemplate="<b>%{x}</b><br>Cumulative: %{y:,}<extra></extra>",
            name="Cumulative",
        ))
        apply(fig3, h=300, title_text="Cumulative FEMA Declarations Over Time")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="sec-head">04 — DECADE-WISE COMPARISON</div>', unsafe_allow_html=True)
    dec_df = flt.groupby("decade").size().reset_index(name="count").sort_values("decade")
    fig4 = go.Figure(go.Bar(
        x=dec_df["decade"], y=dec_df["count"],
        marker=dict(
            color=dec_df["count"],
            colorscale=[[0,"#122a52"],[0.5,"#ff6b35"],[1,"#ff3860"]],
            showscale=False,
            line=dict(color="#ff6b35", width=0.5),
        ),
        text=dec_df["count"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        textfont=dict(color=TEXT_HI, size=11),
        hovertemplate="<b>%{x}</b><br>Declarations: %{y:,}<extra></extra>",
    ))
    apply(fig4, h=300, title_text="Total Disaster Declarations by Decade")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="sec-head">05 — MONTHLY SEASONALITY</div>', unsafe_allow_html=True)
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mon = flt.groupby("month_name").size().reset_index(name="count")
    mon["month_name"] = pd.Categorical(mon["month_name"], categories=month_order, ordered=True)
    mon = mon.sort_values("month_name")

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=mon["month_name"], y=mon["count"],
        marker=dict(color=mon["count"], colorscale="Oranges", showscale=False),
        text=mon["count"].apply(lambda x: f"{x:,}"),
        textposition="outside", textfont=dict(color=TEXT_HI, size=10),
        hovertemplate="<b>%{x}</b><br>%{y:,} declarations<extra></extra>",
    ))
    apply(fig5, h=300, title_text="Disaster Declarations by Month — Seasonality Pattern")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="sec-head">06 — QUARTERLY PATTERN (POLAR)</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        q_df = flt.groupby("quarter").size().reset_index(name="count")
        q_df["quarter"] = "Q" + q_df["quarter"].astype(str)
        fig6 = go.Figure(go.Barpolar(
            r=q_df["count"], theta=q_df["quarter"],
            marker_color=["#ff6b35","#00c9ff","#ff3860","#7c5cbf"],
            marker_line_color="#071428", marker_line_width=2,
            opacity=0.85,
            hovertemplate="<b>%{theta}</b><br>%{r:,} declarations<extra></extra>",
        ))
        apply(fig6, h=340, title_text="Quarterly Declarations — Polar Chart",
              polar=dict(
                  bgcolor="rgba(0,0,0,0)",
                  radialaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT_LO)),
                  angularaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT_HI)),
              ))
        st.plotly_chart(fig6, use_container_width=True)

    with c4:
        st.markdown('<div class="sec-head">07 — TOP 6 TYPES TREND</div>', unsafe_allow_html=True)
        top6 = flt["incidentType"].value_counts().head(6).index.tolist()
        trend = (flt[flt["incidentType"].isin(top6)]
                 .groupby(["year","incidentType"]).size().reset_index(name="count"))
        fig7 = px.line(trend, x="year", y="count", color="incidentType",
                       color_discrete_sequence=PALETTE,
                       labels={"year":"Year","count":"Declarations","incidentType":"Type"})
        fig7.update_traces(line=dict(width=2))
        apply(fig7, h=340, title_text="Top 6 Incident Types Over Time")
        st.plotly_chart(fig7, use_container_width=True)
    st.markdown('<div class="sec-head">08 — FUTURE DISASTER FORECAST</div>', unsafe_allow_html=True)

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["count"],
        name="Actual"
    ))

    if len(yearly) > 5:
        fig_pred.add_trace(go.Scatter(
            x=future_df["year"], y=future_df["predicted"],
            name="Forecast",
            line=dict(dash="dash")
        ))

    apply(fig_pred, h=350, title_text="Future Disaster Risk Forecast")
    st.plotly_chart(fig_pred, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — GEOGRAPHIC
# ─────────────────────────────────────────────────────────────────────────────
with t2:
    st.info("""
        Risk Score is calculated using:
        • 50% historical disaster frequency  
        • 50% recent (last 10 years) activity  

        This helps identify both long-term and emerging high-risk states.
    """)
    # st.dataframe(
    #     risk_df[["risk_score", "risk_level"]].head(10)
    # )
    st.markdown("### 🧾 Top Risk States")

    styled_df = risk_df[["risk_score", "risk_level"]] \
        .sort_values("risk_score", ascending=False) \
        .head(10)

    def color_risk(val):
        if val == "High":
            return "color: red; font-weight: bold;"
        elif val == "Medium":
            return "color: orange; font-weight: bold;"
        else:
            return "color: green; font-weight: bold;"

    st.dataframe(
        styled_df.style.applymap(color_risk, subset=["risk_level"])
    )
    st_counts = flt.groupby("state").size().reset_index(name="count")

    st.markdown('<div class="sec-head">08 — CHOROPLETH MAP — DECLARATIONS BY STATE</div>', unsafe_allow_html=True)
    fig8 = px.choropleth(
        st_counts, locations="state", locationmode="USA-states",
        color="count", color_continuous_scale="YlOrRd", scope="usa",
        labels={"count":"Declarations"},
        hover_data={"count":True},
    )
    fig8.update_layout(
        **BASE_LAYOUT, height=460,
        geo=dict(bgcolor=BG, lakecolor="#071428", landcolor="#0b1e3d",
                 subunitcolor=GRID, showlakes=True, showcoastlines=True, coastlinecolor=GRID),
        coloraxis_colorbar=dict(
            title=dict(text="Count", font=dict(color=TEXT_HI)),
            tickfont=dict(color=TEXT_MID),
            bgcolor="rgba(7,20,40,0.8)", bordercolor=GRID,
        ),
        title_text="Total FEMA Disaster Declarations by US State",
    )
    st.plotly_chart(fig8, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-head">09 — TOP STATES BAR CHART</div>', unsafe_allow_html=True)
        top_st = st_counts.nlargest(top_n, "count").sort_values("count")
        fig9 = go.Figure(go.Bar(
            x=top_st["count"], y=top_st["state"], orientation="h",
            marker=dict(color=top_st["count"],
                        colorscale=[[0,"#122a52"],[0.5,"#ff6b35"],[1,"#ff3860"]],
                        showscale=False),
            text=top_st["count"].apply(lambda x: f"{x:,}"),
            textposition="outside", textfont=dict(color=TEXT_HI, size=10),
            hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>",
        ))
        apply(fig9, h=max(340, top_n*22), title_text=f"Top {top_n} States by Disaster Declarations")
        st.plotly_chart(fig9, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-head">10 — REGIONAL DONUT</div>', unsafe_allow_html=True)
        reg = flt.groupby("region_grp").size().reset_index(name="count").sort_values("count", ascending=False)
        fig10 = go.Figure(go.Pie(
            labels=reg["region_grp"], values=reg["count"],
            hole=0.5, marker=dict(colors=PALETTE, line=dict(color="#071428", width=2)),
            textfont=dict(color=TEXT_HI), pull=[0.05]+[0]*(len(reg)-1),
            hovertemplate="<b>%{label}</b><br>%{value:,} declarations<br>%{percent}<extra></extra>",
        ))
        apply(fig10, h=380, title_text="Disaster Declarations by US Region")
        st.plotly_chart(fig10, use_container_width=True)

    st.markdown('<div class="sec-head">11 — STACKED BAR — INCIDENT MIX PER TOP STATE</div>', unsafe_allow_html=True)
    top10_st = st_counts.nlargest(10,"count")["state"].tolist()
    top6_inc = flt["incidentType"].value_counts().head(6).index.tolist()
    stk = (flt[flt["state"].isin(top10_st) & flt["incidentType"].isin(top6_inc)]
           .groupby(["state","incidentType"]).size().reset_index(name="count"))
    fig11 = px.bar(stk, x="state", y="count", color="incidentType",
                   color_discrete_sequence=PALETTE,
                   labels={"state":"State","count":"Declarations","incidentType":"Type"},
                   barmode="stack")
    apply(fig11, h=380, title_text="Incident Type Composition — Top 10 States")
    st.plotly_chart(fig11, use_container_width=True)

    st.markdown('<div class="sec-head">12 — TREEMAP — STATE × INCIDENT TYPE</div>', unsafe_allow_html=True)
    tree = (flt[flt["state"].isin(top10_st) & flt["incidentType"].isin(top6_inc)]
            .groupby(["region_grp","state","incidentType"]).size().reset_index(name="count"))
    fig12 = px.treemap(tree, path=["region_grp","state","incidentType"], values="count",
                       color="count", color_continuous_scale="Oranges",
                       hover_data={"count":True})
    fig12.update_layout(**BASE_LAYOUT, height=420,
                        title_text="Treemap: Region → State → Incident Type",
                        coloraxis_colorbar=dict(
                            title=dict(text="Count", font=dict(color=TEXT_HI)),
                            tickfont=dict(color=TEXT_MID),
                        ))
    fig12.update_traces(marker=dict(line=dict(color="#071428", width=1)))
    st.plotly_chart(fig12, use_container_width=True)
    st.markdown('<div class="sec-head">NEW — STATE RISK RANKING</div>', unsafe_allow_html=True)

    top_risk = risk_df.head(top_n).sort_values("risk_score")

    fig_risk = go.Figure(go.Bar(
        x=top_risk["risk_score"],
        y=top_risk.index,
        orientation="h"
    ))

    apply(fig_risk, h=350, title_text="Top High-Risk States")
    st.plotly_chart(fig_risk, use_container_width=True)
#     st.markdown('<div class="sec-head">NEW — STATE RISK RANKING</div>', unsafe_allow_html=True)

# top_risk = risk_df.head(top_n).sort_values("risk_score")

# fig_risk = go.Figure(go.Bar(
#     x=top_risk["risk_score"],
#     y=top_risk.index,
#     orientation="h"
# ))

# apply(fig_risk, h=350, title_text="Top High-Risk States")
# st.plotly_chart(fig_risk, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — INCIDENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with t3:
    inc_cnt = flt["incidentType"].value_counts().reset_index()
    inc_cnt.columns = ["incidentType","count"]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-head">13 — INCIDENT FREQUENCY BAR</div>', unsafe_allow_html=True)
        top20 = inc_cnt.head(20)
        fig13 = go.Figure(go.Bar(
            x=top20["count"], y=top20["incidentType"], orientation="h",
            marker=dict(color=top20["count"], colorscale="Reds", showscale=False),
            text=top20["count"].apply(lambda x: f"{x:,}"),
            textposition="outside", textfont=dict(color=TEXT_HI, size=10),
            hovertemplate="<b>%{y}</b>: %{x:,}<extra></extra>",
        ))
        apply(fig13, h=550, title_text="Top 20 Incident Types by Declaration Count")
        st.plotly_chart(fig13, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-head">14 — SUNBURST: DECLARATION TYPE → INCIDENT</div>', unsafe_allow_html=True)
        sun = (flt[flt["incidentType"].isin(top6_inc)]
               .groupby(["declarationType","incidentType"]).size().reset_index(name="count"))
        fig14 = px.sunburst(sun, path=["declarationType","incidentType"], values="count",
                            color="count", color_continuous_scale="YlOrRd",
                            hover_data={"count":True})
        fig14.update_layout(**BASE_LAYOUT, height=550,
                            title_text="Sunburst: Declaration Type → Incident Type",
                            coloraxis_colorbar=dict(
                                title=dict(text="Count", font=dict(color=TEXT_HI)),
                                tickfont=dict(color=TEXT_MID),
                            ))
        st.plotly_chart(fig14, use_container_width=True)

    st.markdown('<div class="sec-head">15 — SEASONAL HEATMAP: INCIDENT TYPE × MONTH</div>', unsafe_allow_html=True)
    top8 = inc_cnt["incidentType"].head(8).tolist()
    heat = (flt[flt["incidentType"].isin(top8)]
            .groupby(["incidentType","month"]).size()
            .reset_index(name="count")
            .pivot(index="incidentType", columns="month", values="count")
            .fillna(0))
    heat.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig15 = px.imshow(heat, color_continuous_scale="YlOrRd", aspect="auto",
                      labels=dict(x="Month", y="Incident Type", color="Count"))
    fig15.update_layout(**BASE_LAYOUT, height=380,
                        title_text="Seasonal Heatmap — Incident Type × Month",
                        coloraxis_colorbar=dict(
                            title=dict(text="Count", font=dict(color=TEXT_HI)),
                            tickfont=dict(color=TEXT_MID),
                        ))
    st.plotly_chart(fig15, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="sec-head">16 — INCIDENT SHARE PIE</div>', unsafe_allow_html=True)
        fig16 = go.Figure(go.Pie(
            labels=inc_cnt.head(9)["incidentType"],
            values=inc_cnt.head(9)["count"],
            hole=0.42,
            marker=dict(colors=PALETTE, line=dict(color="#071428", width=2)),
            textfont=dict(color=TEXT_HI),
            hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
        ))
        apply(fig16, h=380, title_text="Top 9 Incident Types — Share of Total")
        st.plotly_chart(fig16, use_container_width=True)

    with c4:
        st.markdown('<div class="sec-head">17 — DECLARATION TYPE MIX</div>', unsafe_allow_html=True)
        dt_df = flt["declarationType"].value_counts().reset_index()
        dt_df.columns = ["type","count"]
        dt_labels = {"DR":"Major Disaster (DR)","EM":"Emergency (EM)","FM":"Fire Management (FM)"}
        dt_df["label"] = dt_df["type"].map(dt_labels).fillna(dt_df["type"])
        fig17 = go.Figure(go.Pie(
            labels=dt_df["label"], values=dt_df["count"],
            hole=0.42,
            marker=dict(colors=["#ff6b35","#00c9ff","#ff3860"],
                        line=dict(color="#071428", width=2)),
            textfont=dict(color=TEXT_HI),
            hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
        ))
        apply(fig17, h=380, title_text="Declaration Type Distribution (DR / EM / FM)")
        st.plotly_chart(fig17, use_container_width=True)

    st.markdown('<div class="sec-head">18 — VIOLIN: YEARLY COUNTS BY INCIDENT TYPE</div>', unsafe_allow_html=True)
    top5 = inc_cnt["incidentType"].head(5).tolist()
    viol = (flt[flt["incidentType"].isin(top5)]
            .groupby(["year","incidentType"]).size().reset_index(name="count"))
    fig18 = px.violin(viol, x="incidentType", y="count", color="incidentType",
                      box=True, points="outliers",
                      color_discrete_sequence=PALETTE,
                      labels={"incidentType":"Incident Type","count":"Declarations/Year"})
    apply(fig18, h=380, title_text="Distribution of Annual Counts — Top 5 Incident Types", showlegend=False)
    st.plotly_chart(fig18, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — STATISTICAL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with t4:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-head">19 — SCATTER: YEAR vs DECLARATIONS</div>', unsafe_allow_html=True)
        yearly2 = flt.groupby("year").size().reset_index(name="count")
        yearly2["decade"] = (yearly2["year"] // 10 * 10).astype(str) + "s"
        # Manual numpy trendline — no statsmodels needed
        x_vals = yearly2["year"].values
        y_vals = yearly2["count"].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        trend_y = m * x_vals + b
        fig19 = px.scatter(yearly2, x="year", y="count", color="decade",
                           size="count", size_max=22,
                           color_discrete_sequence=PALETTE,
                           labels={"year":"Year","count":"Declarations","decade":"Decade"},
                           hover_data={"year":True,"count":True})
        fig19.add_trace(go.Scatter(
            x=x_vals, y=trend_y,
            mode="lines", name="Linear Trend",
            line=dict(color="#ffffff", width=2, dash="dot"),
            hovertemplate="Trend: %{y:.0f}<extra></extra>",
        ))
        apply(fig19, h=380, title_text="Declarations vs Year — Bubble Scatter with Linear Trend")
        st.plotly_chart(fig19, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-head">20 — BOX PLOT: MONTHLY DISTRIBUTION</div>', unsafe_allow_html=True)
        month_yr = flt.groupby(["year","month_name"]).size().reset_index(name="count")
        month_yr["month_name"] = pd.Categorical(month_yr["month_name"],
                                                 categories=["Jan","Feb","Mar","Apr","May","Jun",
                                                              "Jul","Aug","Sep","Oct","Nov","Dec"],
                                                 ordered=True)
        fig20 = px.box(month_yr, x="month_name", y="count", color="month_name",
                       color_discrete_sequence=PALETTE,
                       labels={"month_name":"Month","count":"Declarations/Year"})
        apply(fig20, h=380, title_text="Monthly Declarations Distribution — Box Plot", showlegend=False)
        st.plotly_chart(fig20, use_container_width=True)

    st.markdown('<div class="sec-head">21 — HEATMAP: YEAR × MONTH DECLARATIONS</div>', unsafe_allow_html=True)
    yr_mon = (flt.groupby(["year","month"]).size()
              .reset_index(name="count")
              .pivot(index="year", columns="month", values="count")
              .fillna(0))
    yr_mon.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig21 = px.imshow(yr_mon, color_continuous_scale="YlOrRd", aspect="auto",
                      labels=dict(x="Month", y="Year", color="Count"))
    fig21.update_layout(**BASE_LAYOUT, height=500,
                        title_text="Year × Month Declaration Heatmap — Full Historical Grid",
                        coloraxis_colorbar=dict(
                            title=dict(text="Count", font=dict(color=TEXT_HI)),
                            tickfont=dict(color=TEXT_MID),
                        ))
    st.plotly_chart(fig21, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="sec-head">22 — FUNNEL: TOP 5 INCIDENT TYPES</div>', unsafe_allow_html=True)
        fnl = inc_cnt.head(5)
        fig22 = go.Figure(go.Funnel(
            y=fnl["incidentType"], x=fnl["count"],
            textinfo="value+percent initial",
            textfont=dict(color=TEXT_HI),
            marker=dict(color=PALETTE[:5], line=dict(color="#071428", width=2)),
            connector=dict(line=dict(color=GRID, width=1)),
            hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>",
        ))
        apply(fig22, h=380, title_text="Funnel Chart — Top 5 Incident Types")
        st.plotly_chart(fig22, use_container_width=True)

    with c4:
        st.markdown('<div class="sec-head">23 — GAUGE: PA PROGRAM COVERAGE</div>', unsafe_allow_html=True)
        pa_rate = flt["paProgramDeclared"].mean() * 100 if "paProgramDeclared" in flt.columns else 0
        ia_rate = flt["iaProgramDeclared"].mean() * 100 if "iaProgramDeclared" in flt.columns else 0
        hm_rate = flt["hmProgramDeclared"].mean() * 100 if "hmProgramDeclared" in flt.columns else 0

        fig23 = make_subplots(rows=1, cols=3, specs=[[{"type":"indicator"},{"type":"indicator"},{"type":"indicator"}]])
        for i, (val, title, color) in enumerate([
            (pa_rate, "Public Assist.", "#ff6b35"),
            (ia_rate, "Indiv. Assist.", "#00c9ff"),
            (hm_rate, "Hazard Mitig.", "#7c5cbf"),
        ], 1):
            fig23.add_trace(go.Indicator(
                mode="gauge+number",
                value=round(val, 1),
                number=dict(suffix="%", font=dict(color=TEXT_HI, size=20, family="Orbitron")),
                title=dict(text=title, font=dict(color=TEXT_MID, size=11)),
                gauge=dict(
                    axis=dict(range=[0,100], tickfont=dict(color=TEXT_LO)),
                    bar=dict(color=color),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor=GRID, borderwidth=1,
                    steps=[dict(range=[0,50], color="#0b1e3d"),
                           dict(range=[50,100], color="#122a52")],
                    threshold=dict(line=dict(color=color, width=2), thickness=0.8, value=val),
                ),
            ), row=1, col=i)
        fig23.update_layout(**BASE_LAYOUT, height=340,
                            title_text="Assistance Program Coverage Rates — Gauge Charts")
        st.plotly_chart(fig23, use_container_width=True)

    st.markdown('<div class="sec-head">24 — WATERFALL: DECADE CHANGE</div>', unsafe_allow_html=True)
    dec2 = flt.groupby("decade").size().reset_index(name="count").sort_values("decade")
    dec2["delta"] = dec2["count"].diff().fillna(dec2["count"])
    dec2["measure"] = ["absolute"] + ["relative"] * (len(dec2)-1)
    fig24 = go.Figure(go.Waterfall(
        x=dec2["decade"], y=dec2["delta"],
        measure=dec2["measure"],
        increasing=dict(marker=dict(color="#00bf72")),
        decreasing=dict(marker=dict(color="#ff3860")),
        totals=dict(marker=dict(color="#ff6b35")),
        connector=dict(line=dict(color=GRID, width=1, dash="dot")),
        text=dec2["delta"].apply(lambda x: f"+{x:,.0f}" if x > 0 else f"{x:,.0f}"),
        textposition="outside", textfont=dict(color=TEXT_HI),
        hovertemplate="<b>%{x}</b><br>Change: %{y:,}<extra></extra>",
    ))
    apply(fig24, h=320, title_text="Waterfall Chart — Decade-over-Decade Change in Declarations")
    st.plotly_chart(fig24, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ASSISTANCE PROGRAMS
# ─────────────────────────────────────────────────────────────────────────────
with t5:
    prog_map = {
        "ihProgramDeclared": "Indiv. & Household",
        "iaProgramDeclared": "Individual Assist.",
        "paProgramDeclared": "Public Assist.",
        "hmProgramDeclared": "Hazard Mitigation",
    }
    prog_map = {k: v for k, v in prog_map.items() if k in flt.columns}

    prog_totals = {v: int(flt[k].sum()) for k, v in prog_map.items()}
    prog_df = pd.DataFrame(list(prog_totals.items()), columns=["Program","Count"])

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-head">25 — PROGRAM OVERVIEW DONUT</div>', unsafe_allow_html=True)
        fig25 = go.Figure(go.Pie(
            labels=prog_df["Program"], values=prog_df["Count"],
            hole=0.55, marker=dict(colors=PALETTE[:4], line=dict(color="#071428", width=2)),
            textfont=dict(color=TEXT_HI),
            hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
        ))
        apply(fig25, h=360, title_text="Assistance Programs — Total Declarations Breakdown")
        st.plotly_chart(fig25, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-head">26 — PROGRAM RATES BY INCIDENT TYPE</div>', unsafe_allow_html=True)
        top12_inc = inc_cnt["incidentType"].head(10).tolist()
        rows = []
        for inc in top12_inc:
            sub = flt[flt["incidentType"] == inc]
            for k, lbl in prog_map.items():
                rows.append({"Incident":inc, "Program":lbl, "Rate":sub[k].mean()*100})
        rate_df = pd.DataFrame(rows)
        fig26 = px.bar(rate_df, x="Incident", y="Rate", color="Program",
                       barmode="group", color_discrete_sequence=PALETTE,
                       labels={"Rate":"Coverage Rate (%)","Incident":"Incident Type"})
        apply(fig26, h=360, title_text="Assistance Program Coverage Rate by Incident Type (%)")
        fig26.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig26, use_container_width=True)

    st.markdown('<div class="sec-head">27 — PROGRAM DECLARATIONS TREND</div>', unsafe_allow_html=True)
    prog_time = []
    for k, lbl in prog_map.items():
        t = flt[flt[k] == 1].groupby("year").size().reset_index(name="count")
        t["Program"] = lbl
        prog_time.append(t)
    pt_df = pd.concat(prog_time)
    fig27 = px.line(pt_df, x="year", y="count", color="Program",
                    color_discrete_sequence=PALETTE,
                    labels={"year":"Year","count":"Declarations","Program":"Program"})
    fig27.update_traces(line=dict(width=2.5))
    apply(fig27, h=360, title_text="Assistance Program Declarations Over Time (Annual Trend)")
    st.plotly_chart(fig27, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="sec-head">28 — STATE × PROGRAM HEATMAP</div>', unsafe_allow_html=True)
        top15_s = st_counts.nlargest(15,"count")["state"].tolist()
        sp = {lbl: flt[flt["state"].isin(top15_s)].groupby("state")[k].sum()
              for k, lbl in prog_map.items()}
        sp_df = pd.DataFrame(sp).loc[top15_s]
        fig28 = px.imshow(sp_df, color_continuous_scale="Blues", aspect="auto",
                          labels=dict(x="Program", y="State", color="Count"))
        fig28.update_layout(**BASE_LAYOUT, height=420,
                            title_text="State × Program Heatmap — Top 15 States",
                            coloraxis_colorbar=dict(
                                title=dict(text="Count", font=dict(color=TEXT_HI)),
                                tickfont=dict(color=TEXT_MID),
                            ))
        st.plotly_chart(fig28, use_container_width=True)

    with c4:
        st.markdown('<div class="sec-head">29 — RADAR: PROGRAM PROFILE BY TOP STATE</div>', unsafe_allow_html=True)
        top6_st = st_counts.nlargest(6,"count")["state"].tolist()
        categories = list(prog_map.values())
        fig29 = go.Figure()
        for i, s in enumerate(top6_st):
            sub = flt[flt["state"] == s]
            vals = [sub[k].mean()*100 for k in prog_map.keys()]
            vals_closed = vals + [vals[0]]
            fig29.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=categories + [categories[0]],
                name=s, fill="toself",
                line=dict(color=PALETTE[i], width=2),
                fillcolor=f"rgba({int(PALETTE[i][1:3],16)},{int(PALETTE[i][3:5],16)},{int(PALETTE[i][5:7],16)},0.08)",
                hovertemplate=f"<b>{s}</b><br>%{{theta}}: %{{r:.1f}}%<extra></extra>",
            ))
        apply(fig29, h=420, title_text="Radar: Assistance Program Profile — Top 6 States",
              polar=dict(
                  bgcolor="rgba(0,0,0,0)",
                  radialaxis=dict(range=[0,100], gridcolor=GRID,
                                  tickfont=dict(color=TEXT_LO), ticksuffix="%"),
                  angularaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT_HI)),
              ))
        st.plotly_chart(fig29, use_container_width=True)

    st.markdown('<div class="sec-head">30 — STACKED AREA: PROGRAM MIX OVER TIME</div>', unsafe_allow_html=True)
    fig30 = go.Figure()
    for i, (k, lbl) in enumerate(prog_map.items()):
        t = flt[flt[k] == 1].groupby("year").size().reset_index(name="count")
        fig30.add_trace(go.Scatter(
            x=t["year"], y=t["count"],
            name=lbl, stackgroup="one", fill="tonexty",
            line=dict(color=PALETTE[i], width=1),
            fillcolor=f"rgba({int(PALETTE[i][1:3],16)},{int(PALETTE[i][3:5],16)},{int(PALETTE[i][5:7],16)},0.5)",
            hovertemplate=f"<b>{lbl}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))
    apply(fig30, h=360, title_text="Stacked Area — Assistance Program Mix Over Time")
    st.plotly_chart(fig30, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#1e3a5f;font-size:0.78rem;font-family:Exo 2,sans-serif;'>"
    "🌪️ FEMA Disaster Intelligence Dashboard · 68,542 records · 1953–2025 · "
    "Built with Streamlit + Plotly · Data: DisasterDeclarationsSummaries.csv"
    "</p>",
    unsafe_allow_html=True,
)
with t6:
    # st.markdown("### 🎯 Risk Insights")

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.subheader("Top High-Risk States")
    #     # st.dataframe(risk_df.head(10))

    # with col2:
    #     st.subheader("Peak Months")
    #     st.bar_chart(flt["month_name"].value_counts())
    with t6:
        st.markdown("### 🎯 Risk Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔥 High Risk States")

            high_risk = risk_df[risk_df["risk_level"] == "High"]

            st.dataframe(
                high_risk.sort_values("risk_score", ascending=False).head(10)
            )

        with col2:
            st.subheader("Peak Months")
            st.bar_chart(flt["month_name"].value_counts())

    st.subheader("Recommendations")
    st.write("""
    - Focus on high-risk states  
    - Prepare for seasonal peaks  
    - Monitor increasing trends  
    """)
  








