import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_clustering(monthly_df, k=3):
    try:
        pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        available = [p for p in pollutants if p in monthly_df.columns]

        city_avg = (
            monthly_df.groupby("City")[available]
            .mean()
            .dropna()
        )

        scaler = StandardScaler()
        X = scaler.fit_transform(city_avg)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        city_avg["Cluster"] = kmeans.fit_predict(X)

        return city_avg.reset_index(), None

    except Exception as e:
        return None, str(e)
