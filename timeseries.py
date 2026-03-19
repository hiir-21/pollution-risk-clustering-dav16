"""
timeseries.py
-------------
Time-series analysis for the monthly city-level DataFrame.
Generates trend plots, seasonal decomposition, and year-wise comparisons.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

CITIES       = ["Ahmedabad", "Chennai", "Delhi", "Kolkata", "Shillong", "Mumbai"]
CITY_COLORS  = {
    "Ahmedabad": "#378ADD",
    "Chennai":   "#1D9E75",
    "Delhi":     "#E24B4A",
    "Kolkata":   "#BA7517",
    "Shillong":  "#534AB7",
    "Mumbai":    "#D85A30",
}


# ── Trend line chart ──────────────────────────────────────────────────────────

def plot_yearly_trend(monthly_df: pd.DataFrame, pollutant: str = "PM2.5") -> go.Figure:
    """
    Annual mean of a given pollutant for each city.
    Shows whether pollution is increasing or decreasing over years.
    """
    if pollutant not in monthly_df.columns:
        return go.Figure().update_layout(title=f"{pollutant} not available")

    yearly = (
        monthly_df.groupby(["City", "Year"])[pollutant]
        .mean()
        .reset_index()
    )

    fig = px.line(
        yearly, x="Year", y=pollutant, color="City",
        markers=True,
        color_discrete_map=CITY_COLORS,
        title=f"Year-wise {pollutant} Trend by City",
        labels={pollutant: f"{pollutant} (µg/m³)", "Year": "Year"},
        height=420,
    )
    fig.update_traces(line_width=2, marker_size=6)
    fig.update_layout(legend_title_text="City", hovermode="x unified")
    return fig


# ── Seasonal pattern (monthly avg across all years) ───────────────────────────

def plot_seasonal_pattern(monthly_df: pd.DataFrame, pollutant: str = "PM2.5") -> go.Figure:
    """
    Average pollutant level by month (1–12) for each city.
    Reveals seasonal spikes — especially winter (Oct–Jan) in northern cities.
    """
    if pollutant not in monthly_df.columns:
        return go.Figure().update_layout(title=f"{pollutant} not available")

    seasonal = (
        monthly_df.groupby(["City", "Month"])[pollutant]
        .mean()
        .reset_index()
    )
    month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    seasonal["Month_Name"] = seasonal["Month"].map(month_labels)

    fig = px.line(
        seasonal, x="Month", y=pollutant, color="City",
        markers=True,
        color_discrete_map=CITY_COLORS,
        title=f"Monthly Seasonal Pattern — {pollutant}",
        labels={pollutant: f"{pollutant} (µg/m³)", "Month": "Month"},
        height=420,
    )
    fig.update_xaxes(
        tickvals=list(range(1, 13)),
        ticktext=list(month_labels.values())
    )
    fig.update_layout(hovermode="x unified")
    return fig


# ── Season comparison boxplot ─────────────────────────────────────────────────

def plot_season_boxplot(monthly_df: pd.DataFrame, pollutant: str = "PM2.5") -> go.Figure:
    """
    Boxplot comparing pollutant distributions across the 4 seasons per city.
    """
    if pollutant not in monthly_df.columns:
        return go.Figure().update_layout(title=f"{pollutant} not available")

    season_order = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
    df = monthly_df[monthly_df["Season"].isin(season_order)].copy()

    fig = px.box(
        df, x="Season", y=pollutant, color="City",
        color_discrete_map=CITY_COLORS,
        category_orders={"Season": season_order},
        title=f"{pollutant} Distribution by Season and City",
        labels={pollutant: f"{pollutant} (µg/m³)"},
        height=450,
    )
    fig.update_layout(legend_title_text="City", boxmode="group")
    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────

def plot_correlation_heatmap(monthly_df: pd.DataFrame, city: str = "Delhi") -> go.Figure:
    """
    Correlation matrix of pollutants for a single city.
    """
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Benzene", "NH3"]
    available  = [p for p in pollutants if p in monthly_df.columns]

    city_df = monthly_df[monthly_df["City"] == city][available].dropna()
    if city_df.empty:
        return go.Figure().update_layout(title=f"No data for {city}")

    corr = city_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Pollutant Correlation Matrix — {city}",
        height=450,
        aspect="auto",
    )
    return fig


# ── Decomposition for one city ────────────────────────────────────────────────

def plot_decomposition(monthly_df: pd.DataFrame,
                       city: str = "Delhi",
                       pollutant: str = "PM2.5") -> go.Figure:
    """
    STL-style decomposition: Observed → Trend → Seasonality → Residual.
    Uses statsmodels additive decomposition on the monthly series.
    """
    if pollutant not in monthly_df.columns:
        return go.Figure().update_layout(title=f"{pollutant} not available")

    city_df = (
        monthly_df[monthly_df["City"] == city]
        .sort_values(["Year", "Month"])
        .groupby(["Year", "Month"])[pollutant]
        .mean()
        .reset_index()
    )
    city_df["Date"] = pd.to_datetime(
        city_df["Year"].astype(str) + "-" + city_df["Month"].astype(str).str.zfill(2) + "-01"
    )
    series = city_df.set_index("Date")[pollutant].dropna()

    if len(series) < 24:
        return go.Figure().update_layout(
            title=f"Not enough data to decompose ({city} — {pollutant})"
        )

    try:
        decomp = seasonal_decompose(series, model="additive", period=12, extrapolate_trend="freq")
    except Exception as e:
        return go.Figure().update_layout(title=f"Decomposition failed: {e}")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Observed", "Trend", "Seasonality", "Residual"])

    components = [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]
    row_colors = ["#378ADD", "#E24B4A", "#1D9E75", "#888780"]

    for i, (comp, color) in enumerate(zip(components, row_colors), start=1):
        fig.add_trace(
            go.Scatter(x=comp.index, y=comp.values, mode="lines",
                       line=dict(color=color, width=1.5), showlegend=False),
            row=i, col=1
        )

    fig.update_layout(
        title=f"Time-Series Decomposition — {city} ({pollutant})",
        height=600,
        hovermode="x unified",
    )
    return fig


# ── AQI bucket bar (if AQI_Bucket column exists) ─────────────────────────────

def plot_aqi_distribution(monthly_df: pd.DataFrame) -> go.Figure:
    """
    If the dataset has an AQI_Bucket column, shows distribution per city.
    """
    if "AQI_Bucket" not in monthly_df.columns or "AQI" not in monthly_df.columns:
        return None

    yearly = (
        monthly_df.groupby(["City", "Year"])["AQI"]
        .mean()
        .reset_index()
    )
    fig = px.bar(
        yearly, x="Year", y="AQI", color="City",
        barmode="group",
        color_discrete_map=CITY_COLORS,
        title="Mean AQI by City and Year",
        labels={"AQI": "Mean AQI"},
        height=420,
    )
    return fig
