"""
clustering.py
-------------
Takes the monthly city-level DataFrame from preprocess.py and applies
K-Means clustering to assign each city a pollution risk category.
Also computes the Elbow curve and Silhouette scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Features used for clustering (city-level mean across all years)
CLUSTER_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Benzene"]

# Risk label assignment — based on cluster centroid PM2.5 level
RISK_COLORS = {
    "High Risk":   "#E24B4A",
    "Medium Risk": "#EF9F27",
    "Low Risk":    "#1D9E75",
}


# ── Feature engineering ───────────────────────────────────────────────────────

def build_city_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates monthly data to one row per city.
    Returns a DataFrame with city-level mean pollutant concentrations.
    """
    available = [f for f in CLUSTER_FEATURES if f in monthly_df.columns]
    city_features = (
        monthly_df.groupby("City")[available]
        .mean()
        .reset_index()
    )
    return city_features


def scale_features(city_features: pd.DataFrame) -> tuple:
    """
    Applies StandardScaler to pollutant columns.
    Returns (scaled_array, scaler, feature_columns).
    """
    feat_cols = [c for c in CLUSTER_FEATURES if c in city_features.columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(city_features[feat_cols])
    return X_scaled, scaler, feat_cols


# ── Elbow + Silhouette ────────────────────────────────────────────────────────

def compute_elbow(X_scaled: np.ndarray, max_k: int = 6) -> dict:
    """
    Runs KMeans for k = 2..max_k. Returns dict with inertia and silhouette scores.
    With only 6 cities, max_k should be ≤ 5.
    """
    results = {"k": [], "inertia": [], "silhouette": []}
    for k in range(2, min(max_k + 1, len(X_scaled))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        if len(set(labels)) > 1:
            results["silhouette"].append(silhouette_score(X_scaled, labels))
        else:
            results["silhouette"].append(0)
    return results


def plot_elbow(elbow_results: dict) -> go.Figure:
    """Returns a Plotly figure showing the Elbow curve + Silhouette scores."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=elbow_results["k"],
        y=elbow_results["inertia"],
        mode="lines+markers",
        name="Inertia (WCSS)",
        line=dict(color="#378ADD", width=2),
        marker=dict(size=8),
        yaxis="y1",
    ))

    fig.add_trace(go.Scatter(
        x=elbow_results["k"],
        y=elbow_results["silhouette"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#1D9E75", width=2, dash="dash"),
        marker=dict(size=8),
        yaxis="y2",
    ))

    fig.update_layout(
        title="Elbow Method + Silhouette Score",
        xaxis=dict(title="Number of Clusters (k)", tickmode="linear"),
        yaxis=dict(title="Inertia (WCSS)", side="left"),
        yaxis2=dict(title="Silhouette Score", side="right", overlaying="y"),
        legend=dict(x=0.6, y=0.95),
        hovermode="x unified",
        height=400,
    )
    return fig


# ── K-Means clustering ────────────────────────────────────────────────────────

def run_kmeans(city_features: pd.DataFrame, X_scaled: np.ndarray, k: int = 3) -> pd.DataFrame:
    """
    Runs KMeans with k clusters.
    Assigns human-readable risk labels based on each cluster's mean PM2.5.
    Returns city_features DataFrame with added 'Cluster' and 'Risk_Label' columns.
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    city_features = city_features.copy()
    city_features["Cluster"] = km.fit_predict(X_scaled)

    # Rank clusters by PM2.5 centroid value → assign risk labels
    pm25_col = "PM2.5" if "PM2.5" in city_features.columns else city_features.columns[1]
    cluster_pm25 = city_features.groupby("Cluster")[pm25_col].mean().sort_values(ascending=False)

    risk_map = {}
    labels = ["High Risk", "Medium Risk", "Low Risk"]
    for i, cluster_id in enumerate(cluster_pm25.index):
        risk_map[cluster_id] = labels[min(i, len(labels) - 1)]

    city_features["Risk_Label"] = city_features["Cluster"].map(risk_map)
    return city_features, km


# ── PCA for 2D visualization ──────────────────────────────────────────────────

def plot_clusters_2d(city_features: pd.DataFrame, X_scaled: np.ndarray) -> go.Figure:
    """
    Projects scaled features to 2D via PCA and plots cluster assignments.
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "City": city_features["City"].values,
        "Risk": city_features["Risk_Label"].values,
    })

    fig = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="Risk",
        text="City",
        color_discrete_map=RISK_COLORS,
        title="City Pollution Risk Clusters (PCA — 2D)",
        labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
                "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)"},
        height=450,
    )
    fig.update_traces(textposition="top center", marker_size=16)
    fig.update_layout(legend_title_text="Risk Category")
    return fig


def plot_cluster_heatmap(city_features: pd.DataFrame) -> go.Figure:
    """
    Heatmap of mean pollutant levels per city, sorted by risk label.
    """
    order = ["High Risk", "Medium Risk", "Low Risk"]
    sorted_df = city_features.set_index("City")
    risk_order = sorted_df["Risk_Label"].map({r: i for i, r in enumerate(order)})
    sorted_df = sorted_df.loc[risk_order.sort_values().index]

    feat_cols = [c for c in CLUSTER_FEATURES if c in sorted_df.columns]
    heat_data = sorted_df[feat_cols]

    # Normalize per column (0–1) for comparability
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-9)

    fig = px.imshow(
        heat_norm,
        text_auto=".2f",
        color_continuous_scale="RdYlGn_r",
        title="Normalized Pollutant Profile by City",
        labels={"color": "Normalized level"},
        height=350,
        aspect="auto",
    )
    fig.update_xaxes(side="bottom")
    return fig


# ── Bar chart of PM2.5 by city ────────────────────────────────────────────────

def plot_pm25_bar(city_features: pd.DataFrame) -> go.Figure:
    """Bar chart of mean PM2.5 per city, colored by risk."""
    df = city_features.sort_values("PM2.5", ascending=False)
    colors = df["Risk_Label"].map(RISK_COLORS)

    fig = go.Figure(go.Bar(
        x=df["City"],
        y=df["PM2.5"].round(2),
        marker_color=colors,
        text=df["PM2.5"].round(1),
        textposition="outside",
    ))
    fig.update_layout(
        title="Mean PM2.5 Concentration by City (µg/m³)",
        xaxis_title="City",
        yaxis_title="PM2.5 (µg/m³)",
        showlegend=False,
        height=400,
    )
    return fig


# ── Master function ───────────────────────────────────────────────────────────

def run_clustering(monthly_df: pd.DataFrame, k: int = 3) -> dict:
    """
    Full clustering pipeline.
    Returns a dict with city_features, figures, and elbow results.
    """
    city_features = build_city_features(monthly_df)
    X_scaled, scaler, feat_cols = scale_features(city_features)
    elbow_results = compute_elbow(X_scaled)
    city_features, km_model = run_kmeans(city_features, X_scaled, k=k)

    return {
        "city_features": city_features,
        "X_scaled": X_scaled,
        "km_model": km_model,
        "elbow_results": elbow_results,
        "feat_cols": feat_cols,
        "fig_elbow": plot_elbow(elbow_results),
        "fig_clusters_2d": plot_clusters_2d(city_features, X_scaled),
        "fig_heatmap": plot_cluster_heatmap(city_features),
        "fig_pm25_bar": plot_pm25_bar(city_features),
    }
