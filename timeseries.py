import plotly.express as px

def plot_trend(monthly_df, pollutant="PM2.5"):
    if pollutant not in monthly_df.columns:
        return None

    yearly = (
        monthly_df.groupby(["City", "Year"])[pollutant]
        .mean()
        .reset_index()
    )

    fig = px.line(
        yearly,
        x="Year",
        y=pollutant,
        color="City",
        title=f"{pollutant} Trend"
    )
    return fig
