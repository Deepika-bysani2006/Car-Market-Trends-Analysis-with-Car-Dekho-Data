"""
app.py  —  Car Market Trends Analysis  |  Streamlit Dashboard
=============================================================
Sections
--------
  Page 1 — Market Overview   (KPIs + price distribution + fuel share)
  Page 2 — Price Analysis    (by fuel, seller, transmission, year)
  Page 3 — Explore Data      (scatter plots + top models + data table)
  Page 4 — Predict Price     (ML prediction form)

HOW TO RUN
----------
    streamlit run dashboard/app.py
"""

import os, sys, warnings, pickle
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_cleaning import get_clean_data
from model import load_model, predict_price, MODEL_PATH

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Market Trends Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BLUE    = "#2563eb"
C_INDIGO  = "#4f46e5"
C_ORANGE  = "#ea580c"
C_GREEN   = "#16a34a"
C_RED     = "#dc2626"
C_TEAL    = "#0d9488"
C_MUTED   = "#6b7280"
C_BG      = "#f8fafc"
C_SURFACE = "#ffffff"
PALETTE   = [C_BLUE, C_INDIGO, C_ORANGE, C_GREEN, C_RED, C_TEAL]

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Background */
  .stApp { background-color: #f8fafc; }

  /* Remove default top padding */
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* KPI card */
  .kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }
  .kpi-label  { font-size: 12px; font-weight: 600; color: #6b7280;
                text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }
  .kpi-value  { font-size: 26px; font-weight: 700; color: #1e293b; line-height: 1.2; }
  .kpi-sub    { font-size: 12px; color: #6b7280; margin-top: 4px; }

  /* Section header */
  .section-header {
    font-size: 18px; font-weight: 700; color: #1e293b;
    border-left: 4px solid #2563eb;
    padding-left: 10px; margin: 24px 0 12px 0;
  }
  .section-caption { font-size: 13px; color: #6b7280; margin-bottom: 14px; }

  /* Chart card */
  .chart-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
  }

  /* Prediction result box */
  .pred-box {
    background: #eff6ff; border: 2px solid #2563eb;
    border-radius: 12px; padding: 24px;
    text-align: center; margin-top: 16px;
  }
  .pred-label { font-size: 14px; color: #1d4ed8; font-weight: 600; }
  .pred-value { font-size: 36px; font-weight: 800; color: #1e40af; }
  .pred-note  { font-size: 12px; color: #6b7280; margin-top: 8px; }

  /* Sidebar */
  [data-testid="stSidebar"] { background-color: #1e293b; }
  [data-testid="stSidebar"] * { color: #f1f5f9 !important; }
  [data-testid="stSidebar"] .stMultiSelect span { background-color: #334155 !important; }

  /* Divider */
  hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

  /* Metric delta override */
  [data-testid="stMetricDelta"] { font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL — cached
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    return get_clean_data()

@st.cache_resource(show_spinner="Loading prediction model …")
def load_pred_model():
    if os.path.exists(MODEL_PATH):
        return load_model()
    return None

df_full  = load_data()
ml_model = load_pred_model()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚗 Car Market Trends")
    st.markdown("*Car Dekho Dataset — India*")
    st.markdown("---")

    st.markdown("### 🔍 Filters")

    sel_fuel = st.multiselect(
        "Fuel Type",
        options=sorted(df_full["Fuel_Type"].unique()),
        default=sorted(df_full["Fuel_Type"].unique()),
    )
    sel_seller = st.multiselect(
        "Seller Type",
        options=sorted(df_full["Seller_Type"].unique()),
        default=sorted(df_full["Seller_Type"].unique()),
    )
    sel_trans = st.multiselect(
        "Transmission",
        options=sorted(df_full["Transmission"].unique()),
        default=sorted(df_full["Transmission"].unique()),
    )
    yr_min, yr_max = int(df_full["Year"].min()), int(df_full["Year"].max())
    sel_year = st.slider("Manufacturing Year", yr_min, yr_max, (yr_min, yr_max))
    pr_min, pr_max = float(df_full["Selling_Price"].min()), float(df_full["Selling_Price"].max())
    sel_price = st.slider("Selling Price (Rs Lakhs)", pr_min, pr_max,
                          (pr_min, pr_max), step=0.10)

    st.markdown("---")
    page = st.radio("📌 Navigate", [
        "📊 Market Overview",
        "💰 Price Analysis",
        "🔎 Explore Data",
        "🤖 Predict Price",
    ])
    st.markdown("---")
    st.caption("301 listings · 9 columns\nPython · Pandas · Plotly · Streamlit")


# ── Apply filters ──────────────────────────────────────────────────────────────
df = df_full[
    df_full["Fuel_Type"].isin(sel_fuel) &
    df_full["Seller_Type"].isin(sel_seller) &
    df_full["Transmission"].isin(sel_trans) &
    df_full["Year"].between(*sel_year) &
    df_full["Selling_Price"].between(*sel_price)
].reset_index(drop=True)

# ── Helper utilities ───────────────────────────────────────────────────────────
def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def section(title, caption=""):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)

def no_data():
    st.info("ℹ️ No data matches the current filters. Please adjust the sidebar.")

def chart_layout(fig):
    """Apply consistent background / margin to every Plotly figure."""
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#1e293b",
        margin=dict(t=36, b=40, l=10, r=10),
        title_font=dict(size=14, color="#1e293b"),
    )
    return fig

# ── Guard: stop if 0 rows ──────────────────────────────────────────────────────
if len(df) == 0 and page != "🤖 Predict Price":
    st.error("⚠️ Your current filters returned 0 results. Please adjust the sidebar.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Market Overview":

    # ── Title ──────────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='color:#1e293b;font-size:28px;margin-bottom:2px;'>"
        "🚗 Car Market Trends Analysis</h1>"
        "<p style='color:#6b7280;font-size:14px;margin-top:0;'>"
        "Used-car & 2-wheeler market analysis · Car Dekho India Dataset</p>",
        unsafe_allow_html=True,
    )

    # Filter badge
    if len(df) < len(df_full):
        st.info(f"📊 Showing **{len(df)} of {len(df_full)}** listings after filters.")
    else:
        st.success(f"📊 Showing all **{len(df_full)}** listings.")

    # ── KPI Row ────────────────────────────────────────────────────────────────
    section("Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Total Listings", f"{len(df):,}", f"{df['Car_Name'].nunique()} unique models")
    with c2:
        kpi("Avg Selling Price",
            f"Rs {df['Selling_Price'].mean():.2f}L",
            f"Median Rs {df['Selling_Price'].median():.2f}L")
    with c3:
        kpi("Avg Present Price",
            f"Rs {df['Present_Price'].mean():.2f}L",
            "Current showroom value")
    with c4:
        kpi("Avg Kms Driven",
            f"{df['Kms_Driven'].mean():,.0f}",
            f"Max {df['Kms_Driven'].max():,} km")
    with c5:
        kpi("Avg Car Age",
            f"{df['Car_Age'].mean():.1f} yrs",
            f"Range {df['Car_Age'].min()}–{df['Car_Age'].max()} yrs")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row A: Price Distribution | Fuel Pie ──────────────────────────────────
    section("Price & Fuel Breakdown")
    colA1, colA2 = st.columns([3, 2], gap="medium")

    with colA1:
        fig = px.histogram(
            df, x="Selling_Price", nbins=35,
            color_discrete_sequence=[C_BLUE],
            title="Selling Price Distribution",
            labels={"Selling_Price": "Selling Price (Rs Lakhs)", "count": "Listings"},
            marginal="box",
        )
        fig.update_layout(bargap=0.06, showlegend=False)
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with colA2:
        fuel_c = df["Fuel_Type"].value_counts().reset_index()
        fuel_c.columns = ["Fuel_Type", "Count"]
        fig = px.pie(
            fuel_c, names="Fuel_Type", values="Count",
            color_discrete_sequence=PALETTE,
            title="Fuel Type Share",
            hole=0.42,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          textfont_size=12)
        fig.update_layout(legend=dict(orientation="h", y=-0.12))
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row B: Seller split | Transmission split ──────────────────────────────
    colB1, colB2 = st.columns(2, gap="medium")

    with colB1:
        sel_c = df["Seller_Type"].value_counts().reset_index()
        sel_c.columns = ["Seller_Type", "Count"]
        fig = px.bar(
            sel_c, x="Seller_Type", y="Count",
            color="Seller_Type",
            color_discrete_sequence=[C_BLUE, C_INDIGO],
            title="Listings by Seller Type",
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,
                          xaxis_title="Seller Type", yaxis_title="Listings")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with colB2:
        tr_c = df["Transmission"].value_counts().reset_index()
        tr_c.columns = ["Transmission", "Count"]
        fig = px.bar(
            tr_c, x="Transmission", y="Count",
            color="Transmission",
            color_discrete_sequence=[C_ORANGE, C_TEAL],
            title="Listings by Transmission",
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,
                          xaxis_title="Transmission", yaxis_title="Listings")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Year Distribution ──────────────────────────────────────────────────────
    section("Manufacturing Year Distribution",
            "Vehicles listed per manufacture year — the dataset is dominated by 2012–2017 models.")
    yr_c = df["Year"].value_counts().sort_index()
    fig = px.bar(
        x=yr_c.index.astype(str), y=yr_c.values,
        color=yr_c.values,
        color_continuous_scale="Blues",
        title="Number of Listings by Manufacture Year",
        labels={"x": "Year", "y": "Listings", "color": "Count"},
    )
    fig.update_layout(coloraxis_showscale=False,
                      xaxis_tickangle=45, xaxis_title="Year", yaxis_title="Listings")
    chart_layout(fig)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Price Analysis":

    st.markdown(
        "<h1 style='color:#1e293b;font-size:26px;'>💰 Price Analysis</h1>"
        "<p style='color:#6b7280;font-size:14px;'>How fuel type, seller, "
        "transmission, and year affect selling price</p>",
        unsafe_allow_html=True,
    )

    # ── Avg price by fuel ──────────────────────────────────────────────────────
    section("Average Selling Price by Fuel Type",
            "Diesel cars average Rs 10L+ because they are predominantly larger vehicles (SUVs, sedans).")
    colF1, colF2 = st.columns(2, gap="medium")

    with colF1:
        fuel_avg = (
            df.groupby("Fuel_Type")["Selling_Price"]
            .agg(Avg="mean", Count="count").round(2).reset_index()
            .sort_values("Avg", ascending=False)
        )
        fig = px.bar(
            fuel_avg, x="Fuel_Type", y="Avg",
            text="Avg", color="Fuel_Type",
            color_discrete_sequence=PALETTE,
            title="Mean Selling Price by Fuel Type",
            labels={"Avg": "Avg Price (Rs L)", "Fuel_Type": "Fuel Type"},
            custom_data=["Count"],
        )
        fig.update_traces(
            texttemplate="Rs %{text:.2f}L", textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg Rs %{y:.2f}L<br>n=%{customdata[0]}<extra></extra>",
        )
        fig.update_layout(showlegend=False, yaxis_title="Avg Price (Rs Lakhs)")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with colF2:
        fig = px.box(
            df, x="Fuel_Type", y="Selling_Price",
            color="Fuel_Type", color_discrete_sequence=PALETTE,
            title="Price Distribution by Fuel Type",
            labels={"Selling_Price": "Selling Price (Rs L)"},
        )
        fig.update_layout(showlegend=False)
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Avg price by seller ────────────────────────────────────────────────────
    section("Average Selling Price by Seller Type",
            "Dealers list higher-priced cars. Individuals mostly sell 2-wheelers.")
    colS1, colS2 = st.columns(2, gap="medium")

    with colS1:
        sel_avg = (
            df.groupby("Seller_Type")["Selling_Price"]
            .agg(Avg="mean", Count="count").round(2).reset_index()
        )
        fig = px.bar(
            sel_avg, x="Seller_Type", y="Avg",
            text="Avg", color="Seller_Type",
            color_discrete_sequence=[C_BLUE, C_INDIGO],
            title="Mean Selling Price by Seller Type",
            labels={"Avg": "Avg Price (Rs L)"},
            custom_data=["Count"],
        )
        fig.update_traces(
            texttemplate="Rs %{text:.2f}L", textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg Rs %{y:.2f}L<br>n=%{customdata[0]}<extra></extra>",
        )
        fig.update_layout(showlegend=False, yaxis_title="Avg Price (Rs Lakhs)")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with colS2:
        fig = px.box(
            df, x="Seller_Type", y="Selling_Price",
            color="Seller_Type",
            color_discrete_sequence=[C_BLUE, C_INDIGO],
            title="Price Distribution by Seller Type",
            labels={"Selling_Price": "Selling Price (Rs L)"},
        )
        fig.update_layout(showlegend=False)
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Avg price by transmission ──────────────────────────────────────────────
    section("Average Selling Price by Transmission",
            "Automatic vehicles command a Rs 5L premium on average.")
    colT1, colT2 = st.columns(2, gap="medium")

    with colT1:
        tr_avg = (
            df.groupby("Transmission")["Selling_Price"]
            .agg(Avg="mean", Count="count").round(2).reset_index()
        )
        fig = px.bar(
            tr_avg, x="Transmission", y="Avg",
            text="Avg", color="Transmission",
            color_discrete_sequence=[C_ORANGE, C_TEAL],
            title="Mean Selling Price by Transmission",
            labels={"Avg": "Avg Price (Rs L)"},
            custom_data=["Count"],
        )
        fig.update_traces(
            texttemplate="Rs %{text:.2f}L", textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg Rs %{y:.2f}L<br>n=%{customdata[0]}<extra></extra>",
        )
        fig.update_layout(showlegend=False, yaxis_title="Avg Price (Rs Lakhs)")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with colT2:
        fig = px.violin(
            df, x="Transmission", y="Selling_Price",
            color="Transmission",
            color_discrete_sequence=[C_ORANGE, C_TEAL],
            box=True, points="outliers",
            title="Price Distribution by Transmission",
            labels={"Selling_Price": "Selling Price (Rs L)"},
        )
        fig.update_layout(showlegend=False)
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Avg price by year (dual-axis) ──────────────────────────────────────────
    section("Average Selling Price by Manufacture Year",
            "Bars = number of listings. Line = average selling price for that year.")
    yr_data = (
        df.groupby("Year")["Selling_Price"]
        .agg(Avg="mean", Count="count").round(2).reset_index()
        .sort_values("Year")
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yr_data["Year"].astype(str), y=yr_data["Count"],
        name="Listings", marker_color=C_BLUE, opacity=0.45,
        yaxis="y",
        hovertemplate="Year %{x} · %{y} listings<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=yr_data["Year"].astype(str), y=yr_data["Avg"],
        name="Avg Price (Rs L)", mode="lines+markers",
        line=dict(color=C_ORANGE, width=2.5), marker=dict(size=8),
        yaxis="y2",
        hovertemplate="Year %{x} · Avg Rs %{y:.2f}L<extra></extra>",
    ))
    fig.update_layout(
        title="Listings Count & Avg Price by Year",
        xaxis=dict(title="Year", tickangle=45),
        yaxis=dict(title="Number of Listings"),
        yaxis2=dict(title="Avg Selling Price (Rs L)", overlaying="y",
                    side="right", showgrid=False),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Avg price by Owner ─────────────────────────────────────────────────────
    section("Average Selling Price by Number of Previous Owners",
            "First-owner cars (Owner=0) command the highest resale price.")
    own_avg = (
        df.groupby("Owner")["Selling_Price"]
        .agg(Avg="mean", Count="count").round(2).reset_index()
    )
    fig = px.bar(
        own_avg, x=own_avg["Owner"].astype(str), y="Avg",
        text="Avg", color="Avg",
        color_continuous_scale="Blues",
        title="Mean Selling Price by Owner Count",
        labels={"x": "Number of Previous Owners", "Avg": "Avg Price (Rs L)"},
        custom_data=["Count"],
    )
    fig.update_traces(
        texttemplate="Rs %{text:.2f}L", textposition="outside",
        hovertemplate="Owner=%{x}<br>Avg Rs %{y:.2f}L<br>n=%{customdata[0]}<extra></extra>",
    )
    fig.update_layout(coloraxis_showscale=False, showlegend=False,
                      xaxis_title="Previous Owners", yaxis_title="Avg Price (Rs Lakhs)")
    chart_layout(fig)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Explore Data":

    st.markdown(
        "<h1 style='color:#1e293b;font-size:26px;'>🔎 Explore Data</h1>"
        "<p style='color:#6b7280;font-size:14px;'>"
        "Scatter plots, top models, correlation, and the raw data table</p>",
        unsafe_allow_html=True,
    )

    # ── Kms vs Price ───────────────────────────────────────────────────────────
    section("Kilometres Driven vs Selling Price",
            "Hover over each dot to see car name, year, and other details.")
    fig = px.scatter(
        df, x="Kms_Driven", y="Selling_Price",
        color="Fuel_Type", hover_name="Car_Name",
        hover_data={"Year": True, "Seller_Type": True,
                    "Kms_Driven": ":,", "Selling_Price": ":.2f"},
        opacity=0.70, size_max=9,
        color_discrete_sequence=PALETTE,
        title=f"Kms Driven vs Selling Price  (r = {df['Kms_Driven'].corr(df['Selling_Price']):.3f})",
        labels={"Kms_Driven": "Kilometres Driven", "Selling_Price": "Selling Price (Rs L)"},
    )
    fig.update_layout(legend_title="Fuel Type")
    chart_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Car Age vs Price ───────────────────────────────────────────────────────
    section("Car Age vs Selling Price")
    fig = px.scatter(
        df, x="Car_Age", y="Selling_Price",
        color="Fuel_Type", hover_name="Car_Name",
        opacity=0.70, size_max=9,
        color_discrete_sequence=PALETTE,
        title=f"Car Age vs Selling Price  (r = {df['Car_Age'].corr(df['Selling_Price']):.3f})",
        labels={"Car_Age": "Car Age (Years)", "Selling_Price": "Selling Price (Rs L)"},
    )
    fig.update_layout(legend_title="Fuel Type")
    chart_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Top models ────────────────────────────────────────────────────────────
    section("Top Car Models by Average Selling Price",
            "Only models with ≥2 listings. Helps identify value-holding models.")
    name_stats = (
        df.groupby("Car_Name")["Selling_Price"]
        .agg(Avg="mean", Count="count").reset_index()
    )
    name_stats = name_stats[name_stats["Count"] >= 2].sort_values("Avg").tail(15)
    if len(name_stats) == 0:
        no_data()
    else:
        fig = px.bar(
            name_stats, x="Avg", y="Car_Name", orientation="h",
            text="Avg", color="Avg",
            color_continuous_scale="Blues",
            title="Top 15 Car Models — Avg Selling Price",
            labels={"Avg": "Avg Selling Price (Rs L)", "Car_Name": ""},
            custom_data=["Count"],
        )
        fig.update_traces(
            texttemplate="Rs %{text:.2f}L", textposition="outside",
            hovertemplate="<b>%{y}</b><br>Avg Rs %{x:.2f}L<br>n=%{customdata[0]}<extra></extra>",
        )
        fig.update_layout(
            coloraxis_showscale=False,
            height=max(360, len(name_stats) * 30),
            margin=dict(t=36, b=40, l=10, r=90),
        )
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────────────
    section("Feature Correlation Heatmap",
            "Values near +1 or -1 mean a strong relationship. "
            "Present_Price is the strongest predictor of Selling_Price (r=0.88).")
    num_cols = ["Selling_Price", "Present_Price", "Kms_Driven",
                "Year", "Car_Age", "Owner", "Price_Depreciation"]
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text:.2f}",
        textfont=dict(size=11), hoverongaps=False,
    ))
    fig.update_layout(
        title="Correlation Heatmap — Numeric Features",
        height=480, margin=dict(t=50, b=20),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Data table ────────────────────────────────────────────────────────────
    section("Filtered Dataset")
    all_c = df.columns.tolist()
    def_c = ["Car_Name", "Year", "Selling_Price", "Present_Price",
             "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission",
             "Owner", "Car_Age"]
    sel_c = st.multiselect("Columns to display",
                           options=all_c,
                           default=[c for c in def_c if c in all_c])
    if sel_c:
        disp = df[sel_c].rename(columns={
            "Car_Name": "Car Name", "Selling_Price": "Selling Price (Rs L)",
            "Present_Price": "Present Price (Rs L)", "Kms_Driven": "Kms Driven",
            "Fuel_Type": "Fuel Type", "Seller_Type": "Seller Type",
            "Price_Depreciation": "Depreciation (Rs L)",
        })
        st.dataframe(disp, use_container_width=True, height=420)
        st.caption(f"{len(disp):,} records · {len(sel_c)} columns")
    else:
        st.warning("Select at least one column.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT PRICE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Price":

    st.markdown(
        "<h1 style='color:#1e293b;font-size:26px;'>🤖 Predict Selling Price</h1>"
        "<p style='color:#6b7280;font-size:14px;'>"
        "Enter vehicle details to get an estimated resale price</p>",
        unsafe_allow_html=True,
    )

    # ── Model status banner ────────────────────────────────────────────────────
    if ml_model is None:
        st.error(
            "⚠️ Trained model not found. "
            "Run `python src/model.py` from the project root first, then reload this page."
        )
    else:
        st.success(
            "✅ Prediction model loaded · "
            "**Linear Regression** · R² = 0.75 · MAE = Rs 1.47L · RMSE = Rs 2.52L"
        )

    st.markdown("---")

    # ── Model info expander ────────────────────────────────────────────────────
    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown("""
**Algorithm:** Linear Regression (scikit-learn)

**Why Linear Regression?**
It is easy to understand, fast to train, and performed better than
Random Forest on this dataset (R² 0.75 vs 0.49).

**Features used:**
| Feature | Type | Description |
|---------|------|-------------|
| Present_Price | Numeric | Current showroom price in Rs Lakhs |
| Kms_Driven | Numeric | Total kilometres driven |
| Owner | Numeric | Number of previous owners |
| Car_Age | Numeric | Years since manufacture (2024 − Year) |
| Fuel_Type | Categorical | Petrol / Diesel / CNG |
| Seller_Type | Categorical | Dealer / Individual |
| Transmission | Categorical | Manual / Automatic |

**Evaluation metrics (on 20% held-out test data):**
| Metric | Value | Meaning |
|--------|-------|---------|
| MAE | Rs 1.47L | Average prediction error |
| RMSE | Rs 2.52L | Penalises large errors more |
| R² | 0.75 | 75% of price variance explained |

**Limitation:** The dataset has 299 rows. Predictions on vehicles
very different from the training data may be less reliable.
        """)

    st.markdown("---")

    # ── Input form ─────────────────────────────────────────────────────────────
    section("Enter Vehicle Details")

    # Fix label visibility — override any dark-theme leakage into main content
    st.markdown("""
    <style>
      /* Force all widget labels in main content to be dark and visible */
      .main .stNumberInput label,
      .main .stSelectbox label,
      div[data-testid="stNumberInput"] label,
      div[data-testid="stSelectbox"] label {
        color: #1e293b !important;
        font-size: 14px !important;
        font-weight: 600 !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # ── Row 1: section headers ────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown(
            "<div style='background:#eff6ff;border-left:4px solid #2563eb;"
            "padding:8px 12px;border-radius:6px;margin-bottom:10px;'>"
            "<span style='font-weight:700;color:#1e40af;font-size:14px;'>"
            "🔢 Numeric Details</span></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<div style='background:#f0fdf4;border-left:4px solid #16a34a;"
            "padding:8px 12px;border-radius:6px;margin-bottom:10px;'>"
            "<span style='font-weight:700;color:#15803d;font-size:14px;'>"
            "📅 Age & Ownership</span></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            "<div style='background:#fef3c7;border-left:4px solid #d97706;"
            "padding:8px 12px;border-radius:6px;margin-bottom:10px;'>"
            "<span style='font-weight:700;color:#92400e;font-size:14px;'>"
            "⚙️ Vehicle Type</span></div>",
            unsafe_allow_html=True,
        )

    # ── Row 2: inputs ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("<p style='color:#1e293b;font-size:13px;font-weight:600;margin-bottom:2px;'>💰 Present Price (Rs Lakhs)</p>", unsafe_allow_html=True)
        inp_present = st.number_input(
            "Present Price (Rs Lakhs)",
            min_value=0.10, max_value=100.0, value=6.0, step=0.10,
            help="Current showroom / market price of the car model.",
            label_visibility="collapsed",
        )
        st.markdown("<p style='color:#1e293b;font-size:13px;font-weight:600;margin-bottom:2px;margin-top:10px;'>🛣️ Kilometres Driven</p>", unsafe_allow_html=True)
        inp_kms = st.number_input(
            "Kilometres Driven",
            min_value=0, max_value=600_000, value=30_000, step=1000,
            help="Total kilometres the vehicle has been driven.",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("<p style='color:#1e293b;font-size:13px;font-weight:600;margin-bottom:2px;'>📅 Year of Manufacture</p>", unsafe_allow_html=True)
        inp_year = st.number_input(
            "Year of Manufacture",
            min_value=1990, max_value=2024, value=2016, step=1,
            help="Year the vehicle was manufactured.",
            label_visibility="collapsed",
        )
        st.markdown("<p style='color:#1e293b;font-size:13px;font-weight:600;margin-bottom:2px;margin-top:10px;'>👤 Number of Previous Owners</p>", unsafe_allow_html=True)
        inp_owner = st.selectbox(
            "Number of Previous Owners",
            options=[0, 1, 2, 3],
            help="0 = first owner (you are the first buyer).",
            label_visibility="collapsed",
        )

    with col3:
        st.markdown("<p style='color:#1e293b;font-size:13px;font-weight:600;margin-bottom:2px;'>⛽ Fuel Type</p>", unsafe_allow_html=True)
        inp_fuel = st.selectbox(
            "Fuel Type",
            options=["Petrol", "Diesel", "CNG"],
            label_visibility="collapsed",
        )
        st.markdown("<p style='color:#1e293b;font-size:13px;font-weight:600;margin-bottom:2px;margin-top:10px;'>🏪 Seller Type</p>", unsafe_allow_html=True)
        inp_seller = st.selectbox(
            "Seller Type",
            options=["Dealer", "Individual"],
            label_visibility="collapsed",
        )
        st.markdown("<p style='color:#1e293b;font-size:13px;font-weight:600;margin-bottom:2px;margin-top:10px;'>⚙️ Transmission</p>", unsafe_allow_html=True)
        inp_trans = st.selectbox(
            "Transmission",
            options=["Manual", "Automatic"],
            label_visibility="collapsed",
        )

    # Derived: Car_Age
    car_age = 2024 - inp_year

    st.markdown(
        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
        f"padding:10px 16px;margin-top:12px;display:inline-block;'>"
        f"<span style='color:{C_MUTED};font-size:13px;'>"
        f"🕰️ Car Age: <b style='color:#1e293b;'>{car_age} year(s)</b>"
        f"&nbsp;&nbsp;(2024 &minus; {inp_year})</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict button ─────────────────────────────────────────────────────────
    predict_clicked = st.button("🔮 Predict Selling Price", type="primary",
                                use_container_width=True)

    if predict_clicked:
        if ml_model is None:
            st.error("Model not available. Run `python src/model.py` first.")
        else:
            predicted = predict_price(
                ml_model,
                present_price=inp_present,
                kms_driven=int(inp_kms),
                owner=int(inp_owner),
                car_age=int(car_age),
                fuel_type=inp_fuel,
                seller_type=inp_seller,
                transmission=inp_trans,
            )

            st.markdown(
                f'<div class="pred-box">'
                f'<div class="pred-label">Estimated Resale / Selling Price</div>'
                f'<div class="pred-value">Rs {predicted:.2f} Lakhs</div>'
                f'<div class="pred-note">'
                f'Input: {inp_fuel} · {inp_trans} · {inp_seller} · '
                f'{inp_owner} prev owner(s) · {int(inp_kms):,} km · Age {car_age} yr'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")
            # Show depreciation context
            depreciation = round(inp_present - predicted, 2)
            dep_pct = round((depreciation / inp_present) * 100, 1) if inp_present > 0 else 0
            c1, c2, c3 = st.columns(3)
            with c1:
                kpi("Present Price", f"Rs {inp_present:.2f}L", "Current showroom value")
            with c2:
                kpi("Predicted Price", f"Rs {predicted:.2f}L", "Estimated resale value")
            with c3:
                kpi("Estimated Depreciation",
                    f"Rs {depreciation:.2f}L",
                    f"{dep_pct:.1f}% of present price")

            st.markdown(
                f"<p style='font-size:12px;color:{C_MUTED};margin-top:10px;'>"
                "⚠️ This is a model estimate based on 299 training records. "
                "Always verify with a professional valuation before buying or selling."
                "</p>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER (all pages)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:12px;padding:8px 0;'>"
    "Car Market Trends Analysis &nbsp;·&nbsp; Car Dekho India Dataset &nbsp;·&nbsp; "
    "Python · Pandas · Scikit-learn · Plotly · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
