import streamlit as st
from preprocess import load_data
from clustering import run_clustering
from timeseries import plot_trend

st.set_page_config(page_title="Pollution Dashboard", layout="wide")

st.title("🌍 Pollution Risk Clustering Dashboard")

# Load data
daily_df, monthly_df, err = load_data()

if err:
    st.error(err)
    st.stop()

# Sidebar
pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
)

k = st.sidebar.slider("Number of Clusters", 2, 4, 3)

# Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Clustering", "Trends"])

# --- Overview ---
with tab1:
    st.subheader("Dataset Overview")
    st.write(monthly_df.head())

# --- Clustering ---
with tab2:
    st.subheader("Clustering Results")

    cluster_df, err = run_clustering(monthly_df, k)

    if err:
        st.error(err)
    else:
        st.dataframe(cluster_df)

# --- Trends ---
with tab3:
    st.subheader("Pollution Trends")

    fig = plot_trend(monthly_df, pollutant)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
