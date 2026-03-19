"""
app.py  —  Pollution Risk Clustering Dashboard
DAV 16 | CSE520 | Ahmedabad University

Flat repo structure (all files in root):
  app.py, preprocess.py, clustering.py, timeseries.py
  city_day.csv  (committed to repo)
  requirements.txt
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Pollution Risk Clustering | DAV 16",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from preprocess import run_pipeline, get_null_report, POLLUTANTS, CITIES
from clustering  import run_clustering, RISK_COLORS
from timeseries  import (
    plot_yearly_trend, plot_seasonal_pattern,
    plot_season_boxplot, plot_correlation_heatmap, plot_decomposition
)

st.markdown("""
<style>
.main-title  { font-size:2.1rem; font-weight:700; color:#1a1a2e; margin-bottom:0; }
.sub-title   { font-size:0.95rem; color:#666; margin-top:2px; }
.risk-high   { background:#FCEBEB; color:#A32D2D; padding:5px 13px; border-radius:20px; font-weight:600; display:inline-block; }
.risk-medium { background:#FAEEDA; color:#854F0B; padding:5px 13px; border-radius:20px; font-weight:600; display:inline-block; }
.risk-low    { background:#EAF3DE; color:#3B6D11; padding:5px 13px; border-radius:20px; font-weight:600; display:inline-block; }
.city-card   { background:#f8f9fa; border-radius:12px; padding:18px 12px; text-align:center; border:1px solid #e4e4e4; margin-bottom:8px; }
.section-hdr { font-size:1.2rem; font-weight:600; color:#1a1a2e; margin:1.4rem 0 0.4rem; border-bottom:2px solid #ebebeb; padding-bottom:4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists("city_day.csv"):
        try:
            daily_df, monthly_df = run_pipeline("city_day.csv")
            return daily_df, monthly_df, None
        except Exception as e:
            return None, None, str(e)
    if os.path.exists("monthly_city_data.csv"):
        try:
            monthly_df = pd.read_csv("monthly_city_data.csv")
            season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Pre-Monsoon",4:"Pre-Monsoon",
                          5:"Pre-Monsoon",6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
                          10:"Post-Monsoon",11:"Post-Monsoon"}
            if "Season" not in monthly_df.columns:
                monthly_df["Season"] = monthly_df["Month"].map(season_map)
            return monthly_df, monthly_df, None
        except Exception as e:
            return None, None, str(e)
    return None, None, "city_day.csv not found. Ensure it is committed to the repository root."


@st.cache_data(show_spinner=False)
def get_clusters(_monthly_df, k):
    return run_clustering(_monthly_df, k=k)


with st.sidebar:
    st.markdown("## 🌫️ Pollution Risk Clustering")
    st.caption("DAV 16 · CSE520 · Ahmedabad University")
    st.divider()
    st.markdown("### Clustering")
    k_clusters = st.slider("Number of clusters (k)", 2, 5, 3)
    st.markdown("### Chart options")
    sel_pollutant = st.selectbox("Pollutant", ["PM2.5","PM10","NO2","SO2","CO","O3","Benzene"], index=0)
    sel_city      = st.selectbox("City (decomposition)", CITIES, index=2)
    st.divider()
    st.markdown("**Team — DAV 16**")
    for name in ["Hiir Jadav","Preet Kaur","Aarya Gogia","Vrushank Thakkar"]:
        st.markdown(f"· {name}")


st.markdown('<p class="main-title">🌫️ Pollution Risk Clustering — Indian Air Quality</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">CPCB · city_day.csv · 2015–2020 · Ahmedabad · Chennai · Delhi · Kolkata · Shillong · Mumbai</p>', unsafe_allow_html=True)
st.divider()

with st.spinner("Loading and preprocessing data..."):
    daily_df, monthly_df, err = load_data()

if err:
    st.error(f"Data load failed: {err}")
    st.stop()

avail = [p for p in ["PM2.5","PM10","NO2","SO2","CO","O3","Benzene","NH3","NOx","NO"] if p in monthly_df.columns]
sel_pollutant = sel_pollutant if sel_pollutant in avail else avail[0]

with st.spinner("Running K-Means..."):
    cr = get_clusters(monthly_df, k_clusters)
city_feat = cr["city_features"]


tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview","🧹 Preprocessing","📈 Trends & EDA","🔵 Clustering","📉 Decomposition"])

with tab1:
    st.markdown('<p class="section-hdr">Dataset Summary</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Daily records", f"{len(daily_df):,}")
    with c2: st.metric("Cities", monthly_df["City"].nunique())
    with c3:
        yrs = sorted(monthly_df["Year"].unique())
        st.metric("Year range", f"{min(yrs)} – {max(yrs)}")
    with c4: st.metric("Pollutants", len(avail))

    st.markdown('<p class="section-hdr">Risk Classification Results</p>', unsafe_allow_html=True)
    cols = st.columns(len(city_feat))
    for i, row in city_feat.iterrows():
        with cols[i % len(cols)]:
            risk = row["Risk_Label"]
            css  = "risk-high" if "High" in risk else ("risk-medium" if "Medium" in risk else "risk-low")
            pm   = row.get("PM2.5", None)
            pm_s = f"{pm:.1f} µg/m³" if pd.notna(pm) else "N/A"
            st.markdown(f'<div class="city-card"><h3 style="margin:0 0 6px">{row["City"]}</h3>'
                        f'<span class="{css}">{risk}</span>'
                        f'<p style="margin:8px 0 0;font-size:0.82rem;color:#666">PM2.5: {pm_s}</p></div>',
                        unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">WHO PM2.5 Reference</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Category":["WHO Guideline","Low Risk","Medium Risk","High Risk"],
        "PM2.5 (µg/m³)":[5,"< 15","15–35","> 35"],
        "Health note":["Annual mean target","Relatively clean","Moderate concern","Significant health risk"]
    }), hide_index=True, use_container_width=True)

with tab2:
    st.markdown('<p class="section-hdr">Shape & Coverage</p>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Daily rows (6 cities)", f"{len(daily_df):,}")
    with c2: st.metric("Monthly aggregated rows", f"{len(monthly_df):,}")
    with c3: st.metric("Columns", len(daily_df.columns))

    st.markdown('<p class="section-hdr">Missing Values After Cleaning</p>', unsafe_allow_html=True)
    st.dataframe(get_null_report(daily_df), hide_index=True, use_container_width=True)

    st.markdown('<p class="section-hdr">Steps Applied</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        ("1","Load city_day.csv","Filter to 6 cities"),
        ("2","Date parsing","Date → datetime, extract Year/Month/Season"),
        ("3","Type coercion","Pollutant columns → numeric; invalid → NaN"),
        ("4","IQR outlier removal","Per-city IQR; extremes → NaN"),
        ("5","Linear interpolation","Fill short gaps ≤7 days per city"),
        ("6","Seasonal mean fill","Remaining NaNs → city × season average"),
        ("7","Monthly aggregation","Daily → monthly city-level means"),
    ], columns=["Step","Operation","Description"]), hide_index=True, use_container_width=True)

    st.markdown('<p class="section-hdr">Sample Monthly Data</p>', unsafe_allow_html=True)
    st.dataframe(monthly_df.head(24), hide_index=True, use_container_width=True)

with tab3:
    st.markdown('<p class="section-hdr">Year-wise Trend</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_yearly_trend(monthly_df, sel_pollutant), use_container_width=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<p class="section-hdr">Monthly Seasonal Pattern</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_seasonal_pattern(monthly_df, sel_pollutant), use_container_width=True)
    with cb:
        st.markdown('<p class="section-hdr">Season-wise Distribution</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_season_boxplot(monthly_df, sel_pollutant), use_container_width=True)
    st.markdown(f'<p class="section-hdr">Pollutant Correlation — {sel_city}</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_correlation_heatmap(monthly_df, sel_city), use_container_width=True)

with tab4:
    st.markdown('<p class="section-hdr">Elbow Curve</p>', unsafe_allow_html=True)
    st.plotly_chart(cr["fig_elbow"], use_container_width=True)
    st.caption("The elbow in inertia + silhouette peak together indicate optimal k.")
    cl, cr2 = st.columns(2)
    with cl:
        st.markdown('<p class="section-hdr">Cluster Plot (PCA 2D)</p>', unsafe_allow_html=True)
        st.plotly_chart(cr["fig_clusters_2d"], use_container_width=True)
    with cr2:
        st.markdown('<p class="section-hdr">PM2.5 by City</p>', unsafe_allow_html=True)
        st.plotly_chart(cr["fig_pm25_bar"], use_container_width=True)
    st.markdown('<p class="section-hdr">Normalized Pollutant Heatmap</p>', unsafe_allow_html=True)
    st.plotly_chart(cr["fig_heatmap"], use_container_width=True)
    disp = ["City","Risk_Label"] + [c for c in ["PM2.5","PM10","NO2","SO2","CO"] if c in city_feat.columns]
    st.dataframe(city_feat[disp].rename(columns={"Risk_Label":"Risk Category"})
                 .sort_values("Risk Category").reset_index(drop=True),
                 hide_index=True, use_container_width=True)

with tab5:
    st.markdown(f'<p class="section-hdr">Decomposition — {sel_city} ({sel_pollutant})</p>', unsafe_allow_html=True)
    st.caption("Additive: Observed = Trend + Seasonality + Residual")
    st.plotly_chart(plot_decomposition(monthly_df, sel_city, sel_pollutant), use_container_width=True)
    st.dataframe(pd.DataFrame([
        ("Observed","Raw monthly data"),
        ("Trend","Long-term rise or fall in pollution"),
        ("Seasonality","Recurring annual cycle (winter spikes Oct–Jan)"),
        ("Residual","Unexplained variation / noise"),
    ], columns=["Component","Meaning"]), hide_index=True, use_container_width=True)

st.divider()
st.caption("DAV 16 · Hiir Jadav · Preet Kaur · Aarya Gogia · Vrushank Thakkar · CSE520 · Ahmedabad University · 2025")
