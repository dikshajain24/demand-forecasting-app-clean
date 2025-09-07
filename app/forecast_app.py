# app/forecast_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path

# Paths
PROCESSED = Path("data/processed")
MODELS = Path("models")

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

st.title("📈 Demand Forecasting Dashboard")

# --- Load datasets ---
@st.cache_data
def load_processed():
    path = PROCESSED / "features.parquet"
    if not path.exists():
        st.error("Processed dataset not found. Run ETL + feature scripts first.")
        return None
    return pd.read_parquet(path)

@st.cache_data
def load_predictions(file_name):
    path = MODELS / file_name
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_metrics():
    path = MODELS / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

# Sidebar controls
st.sidebar.header("⚙️ Controls")

# Choose forecast horizon
forecast_file = st.sidebar.selectbox(
    "Select forecast horizon:",
    ["next_7day_preds.csv", "next_14day_preds.csv"]
)

# Load data
hist = load_processed()
preds = load_predictions(forecast_file)
metrics = load_metrics()

# --- Show model metrics ---
if metrics:
    st.sidebar.subheader("📊 Model Performance")
    st.sidebar.metric("Validation RMSE", f"{metrics['Validation_RMSE']:.2f}")
    st.sidebar.metric("Validation MAPE", f"{metrics['Validation_MAPE']:.2f}%")

if hist is not None and preds is not None:
    # Store selector
    store_ids = sorted(hist["store"].unique())
    store_id = st.sidebar.selectbox("Select store:", store_ids)

    # Filter historical data
    store_hist = hist[hist["store"] == store_id].sort_values("date")

    # --- Forecast for selected store (fix: align types) ---
    preds["store"] = preds["store"].astype(str)
    store_pred = preds[preds["store"] == str(store_id)].reset_index(drop=True)

    # Plot historical + forecast
    fig = go.Figure()

    # Historical sales
    fig.add_trace(go.Scatter(
        x=store_hist["date"],
        y=store_hist["sales"],
        mode="lines",
        name="Historical Sales"
    ))

    # Forecast
    if not store_pred.empty:
        last_date = store_hist["date"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=store_pred.shape[1]-1, freq="D")
        forecast_values = store_pred.iloc[0, 1:].values

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_values,
            mode="lines+markers",
            name="Forecast"
        ))

    fig.update_layout(
        title=f"Store {store_id} — Historical vs Forecast",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Show forecast table (fix: transpose for readability) ---
    if not store_pred.empty:
        forecast_tidy = store_pred.drop(columns=["store"]).T.reset_index()
        forecast_tidy.columns = ["Day", "Predicted Sales"]

        st.subheader("📊 Forecasted Values")
        st.dataframe(forecast_tidy)

        # --- Download button ---
        csv = forecast_tidy.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Forecast CSV",
            data=csv,
            file_name=f"store_{store_id}_forecast.csv",
            mime="text/csv"
        )

else:
    st.warning("Please run training + prediction first to generate forecast files.")
