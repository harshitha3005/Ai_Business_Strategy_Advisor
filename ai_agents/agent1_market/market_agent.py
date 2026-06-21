"""
market_agent.py
----------------
Market Research AI Agent (Agent 1) for the AI Business Strategy Advisor.
"""

import json
import warnings

import pandas as pd

from .forecasting import forecast_category

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
USD_TO_INR = 83.0


def to_inr(usd_amount):
    """Convert USD to INR."""
    return round(usd_amount * USD_TO_INR, 2)


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
_object_cols = [
    "id",
    "entity_type",
    "category_code",
    "status",
    "founded_at",
    "country_code",
    "funding_rounds",
    "funding_total_usd",
]

companies = pd.read_csv(
    f"{DATA_DIR}/startup_objects.csv",
    usecols=_object_cols,
    low_memory=False,
)

companies = companies[companies["entity_type"] == "Company"].copy()

companies["category_code"] = companies["category_code"].fillna("unknown")

companies["funding_total_usd"] = pd.to_numeric(
    companies["funding_total_usd"],
    errors="coerce"
)

companies["founded_at"] = pd.to_datetime(
    companies["founded_at"],
    errors="coerce"
)

# ---------------------------------------------------------------------------
# FUNDING DATA
# ---------------------------------------------------------------------------
_funding_cols = [
    "funding_round_id",
    "object_id",
    "funded_at",
    "raised_amount_usd"
]

funding = pd.read_csv(
    f"{DATA_DIR}/startup_funding_rounds.csv",
    usecols=_funding_cols,
    low_memory=False,
)

funding["funded_at"] = pd.to_datetime(
    funding["funded_at"],
    errors="coerce"
)

funding["raised_amount_usd"] = pd.to_numeric(
    funding["raised_amount_usd"],
    errors="coerce"
)

funding["year"] = funding["funded_at"].dt.year

funding = funding.merge(
    companies[["id", "category_code"]],
    left_on="object_id",
    right_on="id",
    how="left"
).drop(columns=["id"])

funding = funding.dropna(
    subset=["year", "category_code", "raised_amount_usd"]
)

funding = funding[
    (funding["year"] >= 2000)
    &
    (funding["year"] <= 2013)
]

# ---------------------------------------------------------------------------
# PIVOT TABLE
# ---------------------------------------------------------------------------
category_year = (
    funding.groupby(["category_code", "year"])["raised_amount_usd"]
    .sum()
    .reset_index()
)

pivot = category_year.pivot(
    index="category_code",
    columns="year",
    values="raised_amount_usd"
).fillna(0)

# ---------------------------------------------------------------------------
# CATEGORY MAPPING
# ---------------------------------------------------------------------------
ALL_CATEGORIES = sorted(pivot.index.unique().tolist())

KEYWORD_TO_CATEGORY = {
    "food": "hospitality",
    "delivery": "hospitality",
    "restaurant": "hospitality",
    "fintech": "finance",
    "finance": "finance",
    "education": "education",
    "learning": "education",
    "health": "health",
    "fitness": "health",
    "software": "software",
    "app": "mobile",
    "mobile": "mobile",
    "game": "games_video",
    "travel": "travel",
    "shopping": "ecommerce",
    "ecommerce": "ecommerce",
}


def match_category(keyword: str):
    keyword = keyword.lower()

    for k, cat in KEYWORD_TO_CATEGORY.items():
        if k in keyword and cat in ALL_CATEGORIES:
            return cat

    for cat in ALL_CATEGORIES:
        if cat in keyword:
            return cat

    return "software"


# ---------------------------------------------------------------------------
# MAIN AGENT OUTPUT (CLEAN)
# ---------------------------------------------------------------------------
def agent_market(keyword: str):

    category = match_category(keyword)

    stats = forecast_category(pivot, category)

    if stats is None:
        stats = {
            "growth_pct": 0.0,
            "forecast_2026": 0.0
        }

    return {
        "agent": "Market Research Agent",
        "input_keyword": keyword,
        "industry_growth_pct": stats["growth_pct"],
        "startup_funding_trend_2026_inr": to_inr(stats["forecast_2026"]),
        "recommended_sector": category
    }