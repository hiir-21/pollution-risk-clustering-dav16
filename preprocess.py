import pandas as pd

CITIES = ["Ahmedabad", "Delhi", "Mumbai", "Chennai", "Kolkata", "Shillong"]

def load_data():
    try:
        df = pd.read_csv("city_day.csv")

        # Filter cities
        df = df[df["City"].isin(CITIES)]

        # Convert date
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month

        # Pollutants
        pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        available = [p for p in pollutants if p in df.columns]

        df = df[["City", "Date", "Year", "Month"] + available]

        # 🔥 STEP 1: Interpolate
        df[available] = df[available].interpolate(limit_direction="both")

        # 🔥 STEP 2: Fill remaining NaN with city-wise mean
        for col in available:
            df[col] = df.groupby("City")[col].transform(
                lambda x: x.fillna(x.mean())
            )

        # 🔥 STEP 3: Final fallback (global mean)
        df[available] = df[available].fillna(df[available].mean())

        # Monthly aggregation
        monthly_df = (
            df.groupby(["City", "Year", "Month"])[available]
            .mean()
            .reset_index()
        )

        return df, monthly_df, None

    except Exception as e:
        return None, None, str(e)
