"""
forecasting.py
---------------
Forecasting and model-evaluation helpers for the Market Research AI Agent
(Agent 1). Extracted from `notebooks/market_analysis.ipynb`.

These functions operate on a `pivot` DataFrame:
    index   = category_code
    columns = year (float, e.g. 2009.0 ... 2013.0)
    values  = total raised_amount_usd for that category/year
"""

import numpy as np
import pandas as pd


# -----------------------------
# PROPHET FORECAST (NEW)
# -----------------------------
def prophet_forecast(pivot, category_code, target_year=2026):

    from prophet import Prophet

    if category_code not in pivot.index:
        return None

    series = pivot.loc[category_code]
    series = series[series > 0]

    if len(series) < 3:
        return None

    df = pd.DataFrame({
        "ds": pd.to_datetime(series.index.astype(int), format="%Y"),
        "y": series.values
    })

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=10, freq="Y")
    forecast = model.predict(future)

    pred = forecast[forecast["ds"].dt.year == target_year]["yhat"].values

    if len(pred) == 0:
        return None

    return {
        "growth_pct": float((df["y"].iloc[-1] / df["y"].iloc[0] - 1) * 100),
        "forecast_2026": float(pred[0]),
    }


# -----------------------------
# ORIGINAL FORECAST FUNCTION (UPDATED)
# -----------------------------
def forecast_category(pivot, category_code, target_year=2026):

    # Try Prophet first
    result = prophet_forecast(pivot, category_code, target_year)
    if result is not None:
        return result

    # fallback to original linear model
    if category_code not in pivot.index:
        return None

    series = pivot.loc[category_code]
    series = series[series > 0]

    if len(series) < 3:
        return None

    years = np.array(series.index, dtype=float)
    values = np.array(series.values, dtype=float)

    smoothed = pd.Series(values).rolling(window=3, min_periods=1).mean()

    coeffs = np.polyfit(years, smoothed, 1)
    forecast = np.polyval(coeffs, target_year)
    forecast = max(float(forecast), 0)

    n_years = years[-1] - years[0]
    cagr = ((values[-1] / values[0]) ** (1 / n_years) - 1) * 100

    return {
        "growth_pct": round(float(cagr), 2),
        "forecast_2026": round(forecast, 2),
    }


# -----------------------------
# EXISTING UTILITIES (UNCHANGED)
# -----------------------------
def calculate_mae(actual, predicted):
    """Mean Absolute Error between two equal-length sequences."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs(actual - predicted))


def calculate_r2(actual, predicted):
    """R-squared (coefficient of determination)."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot == 0:
        return 0

    return 1 - (ss_res / ss_tot)


def evaluate_category(pivot, category_code):

    if category_code not in pivot.index:
        return None

    series = pivot.loc[category_code]
    series = series[series > 0]

    if len(series) < 6:
        return {"mae": 0, "r2": 0}

    years = np.array(series.index, dtype=float)
    values = np.array(series.values, dtype=float)

    maes = []

    for split in range(4, len(values)):

        train_years = years[:split]
        train_values = values[:split]

        test_year = years[split]
        actual = values[split]

        coeffs = np.polyfit(train_years, train_values, 1)
        pred = np.polyval(coeffs, test_year)

        maes.append(abs(actual - pred))

    mae = np.mean(maes)

    return {
        "mae": round(float(mae), 2),
        "r2": "WalkForward",
    }