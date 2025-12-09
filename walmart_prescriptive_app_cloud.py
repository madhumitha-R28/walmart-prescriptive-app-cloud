import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import math
from datetime import timedelta
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Walmart Sales - Prescriptive Analytics (Cloud)",
    layout="wide"
)

st.title("Walmart Sales - Prescriptive Analytics")
st.write(
    "This app loads an engineered Walmart sales dataset and a trained model, "
    "forecasts future sales, and generates simple prescriptive recommendations "
    "like reorder quantities and promotion flags."
)

# ---------- Helper functions ----------

def detect_date_col(df: pd.DataFrame):
    for c in df.columns:
        if c.lower() in ("date", "week", "week_date", "weekdate"):
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None

def find_stock_col(df: pd.DataFrame):
    for c in df.columns:
        cl = c.lower()
        if "stock" in cl or "inventory" in cl or "onhand" in cl:
            return c
    return None

def load_engineered(engineered_path: str) -> pd.DataFrame:
    if not os.path.exists(engineered_path):
        raise FileNotFoundError(f"Engineered data not found at {engineered_path}.")
    df = pd.read_csv(engineered_path)
    date_col = detect_date_col(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.rename(columns={date_col: "Date"})
    else:
        raise ValueError("Could not detect a date column.")
    return df

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def iterative_forecast(df_hist: pd.DataFrame,
                       model,
                       feature_cols,
                       horizon: int = 4,
                       date_freq: str = "W-SUN") -> pd.DataFrame:
    hist = df_hist.copy().sort_values("Date").reset_index(drop=True)
    if "Sales" not in hist.columns:
        raise ValueError("Expected a 'Sales' column in the engineered dataset.")
    sales_series = hist["Sales"].tolist()
    dates = pd.to_datetime(hist["Date"])
    if len(dates) == 0:
        raise ValueError("No dates found in engineered dataset.")
    last_date = dates.max()
    try:
        if len(dates) >= 2:
            step = dates.iloc[-1] - dates.iloc[-2]
        else:
            step = pd.Timedelta(days=7)
    except Exception:
        step = pd.Timedelta(days=7)

    forecasts = []

    for h in range(1, horizon + 1):
        new_date = last_date + h * step
        # extend sales series with prior forecasts
        series_ext = sales_series.copy()
        for f in forecasts:
            series_ext.append(f["Pred"])

        def get_lag(n: int):
            return series_ext[-n] if n <= len(series_ext) else 0.0

        last_row = hist.iloc[-1:]
        new_features = {}
        for col in feature_cols:
            lc = col.lower()
            if lc.startswith("lag_"):
                try:
                    lag_n = int(col.split("_")[-1])
                    new_features[col] = get_lag(lag_n)
                except Exception:
                    new_features[col] = 0.0
            elif lc.startswith("roll_mean_"):
                window = None
                parts = col.split("_")
                for p in reversed(parts):
                    if p.isdigit():
                        window = int(p)
                        break
                if window is None:
                    window = 3
                vals = series_ext[-window:]
                new_features[col] = float(np.mean(vals)) if len(vals) > 0 else 0.0
            elif "pct_change" in lc:
                if len(series_ext) >= 2 and series_ext[-2] != 0:
                    new_features[col] = (series_ext[-1] - series_ext[-2]) / float(series_ext[-2])
                else:
                    new_features[col] = 0.0
            else:
                if col in last_row.columns:
                    new_features[col] = float(last_row.iloc[0].get(col, 0.0))
                else:
                    new_features[col] = 0.0

        X_new = pd.DataFrame([new_features], columns=feature_cols).fillna(0).astype(float)
        try:
            pred = model.predict(X_new)[0]
        except Exception:
            # fallback in case the model ignores some columns
            usable_cols = [c for c in feature_cols if c in X_new.columns]
            pred = model.predict(X_new[usable_cols])[0]
        forecasts.append({"Date": new_date, "Pred": float(pred)})

    return pd.DataFrame(forecasts)

def create_prescriptions(
    forecast_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    stock_col: str = None,
    reorder_service_level_z: float = 1.65,
    min_promo_drop_pct: float = 0.10,
) -> pd.DataFrame:
    if "Sales" not in hist_df.columns:
        raise ValueError("Expected 'Sales' column in historical dataframe.")
    last_sales = hist_df["Sales"].iloc[-1]
    prescriptions = []

    for idx, row in forecast_df.iterrows():
        pred = row["Pred"]
        date = row["Date"]
        if idx == 0:
            prev = last_sales
        else:
            prev = forecast_df.loc[idx - 1, "Pred"]
        pct_change = (pred - prev) / prev if prev != 0 else 0.0

        # Promotion rule
        promo = False
        promo_reason = ""
        if pct_change < -min_promo_drop_pct:
            promo = True
            promo_reason = f"Forecast drop {pct_change:.1%} — consider promotions."

        # Reorder rule
        if stock_col and stock_col in hist_df.columns:
            demand_std = hist_df["Sales"].std()
            safety = reorder_service_level_z * demand_std
            last_stock_value = hist_df[stock_col].iloc[-1]
            current_stock = last_stock_value if not pd.isna(last_stock_value) else 0.0
            reorder = max(0.0, pred + safety - current_stock)
            reorder_reason = f"Reorder to cover forecast ({pred:.1f}) + safety."
        else:
            reorder = max(pred, 0.0)
            reorder_reason = "No stock data — reorder equals forecast."

        impact = abs(pct_change) * max(pred, 1.0)
        prescriptions.append(
            {
                "Date": date,
                "Forecast": pred,
                "Pct_change": pct_change,
                "Recommend_Promo": promo,
                "Promo_Reason": promo_reason,
                "Reorder_Qty": round(reorder, 2),
                "Reorder_Reason": reorder_reason,
                "Priority_Impact": impact,
            }
        )

    return pd.DataFrame(prescriptions)

# ---------- Streamlit UI ----------

st.sidebar.header("Configuration")

default_engineered_path = "engineered_full.csv"
default_model_path = "final_model.pkl"

st.sidebar.subheader("Data & Model")

engineered_file = st.sidebar.file_uploader(
    "Upload engineered_full.csv (optional)",
    type=["csv"],
    help="If not provided, the app will look for engineered_full.csv in the same folder as this script.",
)

model_file = st.sidebar.file_uploader(
    "Upload final_model.pkl (optional)",
    type=["pkl"],
    help="If not provided, the app will look for final_model.pkl in the same folder as this script.",
)

horizon = st.sidebar.slider("Forecast horizon (periods)", min_value=1, max_value=52, value=12, step=1)
date_freq_label = st.sidebar.selectbox(
    "Date frequency used when the model was trained",
    options=["Weekly (W-SUN)", "Daily (D)"],
    index=0,
)
freq_map = {
    "Weekly (W-SUN)": "W-SUN",
    "Daily (D)": "D",
}
date_freq = freq_map[date_freq_label]

service_level_z = st.sidebar.number_input(
    "Reorder service level Z",
    min_value=0.0,
    max_value=3.0,
    value=1.65,
    step=0.05,
)

min_promo_drop_pct = st.sidebar.number_input(
    "Min % drop to trigger promotion",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01,
    help="E.g. 0.10 = 10% drop vs previous period.",
)

top_n = st.sidebar.number_input(
    "Top N periods by impact to highlight",
    min_value=1,
    max_value=52,
    value=10,
    step=1,
)

run_button = st.sidebar.button("Run Prescriptive Analytics")

def get_engineered_df():
    if engineered_file is not None:
        return pd.read_csv(engineered_file)
    else:
        return load_engineered(default_engineered_path)

def get_model_obj():
    if model_file is not None:
        return pickle.load(model_file)
    else:
        return load_model(default_model_path)

# Auto-run if using local default files (no uploads), OR when button is clicked
if run_button or (engineered_file is None and model_file is None):
    try:
        engineered = get_engineered_df()
        st.success("Engineered dataset loaded.")
    except Exception as e:
        st.error(f"Error loading engineered dataset: {e}")
        st.stop()

    try:
        model = get_model_obj()
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Derive feature columns
    non_feat = set(["Date", "Sales", "Target", "Year", "Quarter"])
    feature_cols = [c for c in engineered.columns if c not in non_feat and engineered[c].dtype != "O"]
    if len(feature_cols) == 0:
        feature_cols = [
            c
            for c in engineered.select_dtypes(include=[np.number]).columns
            if c not in ("Sales", "Target")
        ]

    engineered = engineered.sort_values("Date").reset_index(drop=True)

    # Ensure Date is proper datetime in historical data
    engineered["Date"] = pd.to_datetime(engineered["Date"], errors="coerce")

    st.write(f"Using {len(feature_cols)} feature columns for forecasting.")

    try:
        forecast_df = iterative_forecast(
            engineered,
            model,
            feature_cols,
            horizon=int(horizon),
            date_freq=date_freq,
        )
    except Exception as e:
        st.error(f"Error while generating forecast: {e}")
        st.stop()

    # Ensure Date is proper datetime in forecast data
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors="coerce")

    stock_col = find_stock_col(engineered)

    try:
        prescriptions = create_prescriptions(
            forecast_df,
            engineered,
            stock_col=stock_col,
            reorder_service_level_z=float(service_level_z),
            min_promo_drop_pct=float(min_promo_drop_pct),
        )
    except Exception as e:
        st.error(f"Error while generating prescriptions: {e}")
        st.stop()

    # ---------- Visualization ----------
    st.subheader("Historical vs Forecast")

    # Extra safety: convert to datetime right before plotting
    hist_dates = pd.to_datetime(engineered["Date"], errors="coerce")
    fc_dates = pd.to_datetime(forecast_df["Date"], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist_dates, engineered["Sales"], marker="o", label="History")
    ax.plot(fc_dates, forecast_df["Pred"], marker="x", label="Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Prescriptive Recommendations (All Periods)")
    st.dataframe(prescriptions)

    st.subheader(f"Top {top_n} Periods by Priority_Impact")
    top_table = prescriptions.sort_values("Priority_Impact", ascending=False).head(int(top_n))
    st.dataframe(top_table)

    # ---------- Business Conclusions ----------
    st.subheader("Business Conclusions")

    try:
        # Basic forecast summary
        total_forecast = prescriptions["Forecast"].sum()
        avg_forecast = prescriptions["Forecast"].mean()
        first_forecast = prescriptions["Forecast"].iloc[0]
        last_forecast = prescriptions["Forecast"].iloc[-1]
        trend_pct = ((last_forecast - first_forecast) / first_forecast) * 100 if first_forecast != 0 else 0

        peak_row = prescriptions.loc[prescriptions["Forecast"].idxmax()]
        peak_date = peak_row["Date"]
        peak_value = peak_row["Forecast"]

        drop_rows = prescriptions[prescriptions["Pct_change"] < 0]
        if not drop_rows.empty:
            worst_drop_row = drop_rows.loc[drop_rows["Pct_change"].idxmin()]
            worst_drop_date = worst_drop_row["Date"]
            worst_drop_pct = worst_drop_row["Pct_change"] * 100
        else:
            worst_drop_date = None
            worst_drop_pct = 0

        # Recommendations summary
        total_reorder_qty = prescriptions["Reorder_Qty"].sum()
        promo_periods = prescriptions[prescriptions["Recommend_Promo"]]
        num_promo_periods = len(promo_periods)

        high_impact_periods = top_table.copy()
        high_impact_periods_dates = high_impact_periods["Date"].dt.date.astype(str).tolist()

        st.markdown("### Forecast – What does the model indicate?")
        st.markdown(
            f"""
- **Total forecasted demand** over the selected horizon: **{total_forecast:,.0f} units**  
- **Average weekly/period demand**: **{avg_forecast:,.0f} units**  
- **Demand trend over forecast horizon**: {"increase" if trend_pct > 0 else "decrease" if trend_pct < 0 else "flat"} of **{trend_pct:+.1f}%** from first to last period  
- **Peak forecast period**: **{str(getattr(peak_date, "date", lambda: peak_date)()) if hasattr(peak_date, "date") else str(peak_date)}** with **{peak_value:,.0f} units** expected
"""
        )

        if worst_drop_date is not None:
            st.markdown(
                f"- **Largest expected drop** in demand around **{str(getattr(worst_drop_date, 'date', lambda: worst_drop_date)()) if hasattr(worst_drop_date, 'date') else str(worst_drop_date)}**: approximately **{worst_drop_pct:.1f}%** vs previous period."
            )
        else:
            st.markdown("- No significant forecasted drops in demand across the horizon.")

        st.markdown("### Recommendations – How should the business respond?")
        st.markdown(
            f"""
- **Total recommended reorder quantity** over the horizon: **{total_reorder_qty:,.0f} units**  
- **Number of periods where promotions are recommended**: **{num_promo_periods}**  
"""
        )

        if num_promo_periods > 0:
            promo_dates = promo_periods["Date"].dt.date.astype(str).tolist()
            st.markdown(
                f"- Promotions are mainly recommended in these periods (likely demand softening): `{promo_dates}`"
            )

        st.markdown(
            f"- **High-priority periods** (top {top_n} by impact) to focus inventory & marketing decisions on: `{high_impact_periods_dates}`"
        )

        st.markdown(
            """
**Managerial takeaway:**  
Use the high-demand periods to **ensure sufficient stock and avoid stockouts**, and use the promo-flagged periods to plan **discounts, campaigns, or store-level initiatives** to prevent revenue loss. The reorder recommendations can guide **purchase planning** so that inventory covers forecast + safety stock without overstocking.
"""
        )
    except Exception as e:
        st.warning(f"Could not auto-generate business conclusions: {e}")

    # ---------- Downloads ----------
    st.subheader("Download Outputs")

    forecast_csv = forecast_df.rename(columns={"Pred": "Forecast"}).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast CSV",
        data=forecast_csv,
        file_name="prescriptive_forecast_series.csv",
        mime="text/csv",
    )

    presc_csv = prescriptions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prescriptive recommendations CSV",
        data=presc_csv,
        file_name="prescriptive_recommendations.csv",
        mime="text/csv",
    )

else:
    st.info(
        "Upload custom files in the sidebar and click **Run Prescriptive Analytics**, "
        "or place engineered_full.csv and final_model.pkl next to this script to auto-run."
    )
