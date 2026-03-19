"""
preprocess.py
-------------
Loads city_day.csv (already a merged, city-level daily file from Kaggle).
Filters for 6 target cities, cleans, imputes, and aggregates to monthly level.
"""

import pandas as pd
import numpy as np

CITIES = ["Ahmedabad", "Chennai", "Delhi", "Kolkata", "Shillong", "Mumbai"]

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "O3", "Benzene"]


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Pre-Monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


def load_city_day(csv_path: str = "city_day.csv") -> pd.DataFrame:
    """Loads city_day.csv and filters for our 6 cities."""
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded city_day.csv → {len(df):,} rows, {df['City'].nunique()} cities")

    df = df[df["City"].isin(CITIES)].copy()
    print(f"   After city filter → {len(df):,} rows")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Year"]   = df["Date"].dt.year
    df["Month"]  = df["Date"].dt.month
    df["Season"] = df["Month"].map(get_season)
    return df


def coerce_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in POLLUTANTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in POLLUTANTS if c in df.columns]
    for city, grp in df.groupby("City"):
        for col in cols:
            q1, q3 = grp[col].quantile(0.25), grp[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = df["City"] == city
            df.loc[mask & ((df[col] < lo) | (df[col] > hi)), col] = np.nan
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in POLLUTANTS if c in df.columns]

    # Linear interpolation per city (short gaps)
    df = df.sort_values(["City", "Date"])
    df[cols] = df.groupby("City")[cols].transform(
        lambda s: s.interpolate(method="linear", limit=7)
    )

    # Seasonal mean fill (larger gaps)
    for col in cols:
        seasonal_means = df.groupby(["City", "Season"])[col].transform("mean")
        df[col] = df[col].fillna(seasonal_means)

    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in POLLUTANTS if c in df.columns]
    agg = (
        df.groupby(["City", "Year", "Month", "Season"])[cols]
        .mean()
        .reset_index()
    )
    return agg.sort_values(["City", "Year", "Month"]).reset_index(drop=True)


def get_null_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in POLLUTANTS if c in df.columns]
    return pd.DataFrame({
        "Column":     cols,
        "Null Count": [df[c].isna().sum() for c in cols],
        "Null %":     [(df[c].isna().mean() * 100).round(2) for c in cols],
    }).sort_values("Null %", ascending=False).reset_index(drop=True)


def run_pipeline(csv_path: str = "city_day.csv"):
    """Full pipeline. Returns (daily_df, monthly_df)."""
    print("\n📦 Preprocessing pipeline starting...")
    raw      = load_city_day(csv_path)
    dated    = parse_dates(raw)
    coerced  = coerce_pollutants(dated)
    cleaned  = remove_outliers_iqr(coerced)
    imputed  = impute_missing(cleaned)
    monthly  = aggregate_monthly(imputed)

    print(f"✅ Monthly shape: {monthly.shape}")
    print(f"   Cities : {sorted(monthly['City'].unique())}")
    print(f"   Years  : {sorted(monthly['Year'].unique())}")
    return imputed, monthly


if __name__ == "__main__":
    daily, monthly = run_pipeline("city_day.csv")
    monthly.to_csv("monthly_city_data.csv", index=False)
    print("💾 Saved → monthly_city_data.csv")
