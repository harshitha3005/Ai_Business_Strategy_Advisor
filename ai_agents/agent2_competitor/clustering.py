"""
clustering.py — Competitor Intelligence: Feature Engineering, Random Forest & K-Means
 
Extracted and modularised from competitor_analysis.ipynb.
Used by competitor_agent.py to:
  1. Load & clean the Crunchbase startup dataset
  2. Engineer a 'company_age_years' feature
  3. Train a RandomForestRegressor to score each company's overall strength
  4. Cluster companies into 3 market positions via K-Means:
       Market Leader / Challenger / Niche Player
 
Dataset: data/startup_objects.csv  (Crunchbase companies)
"""
 
import os
import warnings
 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
DATA_DIR = os.environ.get("DATA_DIR", "../../data")
CSV_PATH = os.path.join(DATA_DIR, "startup_objects.csv")
 
# Columns loaded from CSV
LOAD_COLS = [
    "id",
    "entity_type",
    "name",
    "category_code",
    "status",
    "country_code",
    "founded_at",
    "funding_total_usd",
    "funding_rounds",
    "milestones",
    "relationships",
]
 
NUMERIC_COLS = ["funding_total_usd", "funding_rounds", "milestones", "relationships"]
 
# Features used for clustering (after feature engineering)
CLUSTER_FEATURES = [
    "funding_total_usd",
    "funding_rounds",
    "milestones",
    "relationships",
    "company_age_years",
]
 
# Features used to train the Random Forest strength predictor
RF_TRAIN_FEATURES = [
    "funding_rounds",
    "milestones",
    "relationships",
    "company_age_years",
]
 
# Cluster label constants
POSITION_LABELS = ["Market Leader", "Challenger", "Niche Player"]
 
POSITION_EXPLANATIONS = {
    "Market Leader": "Dominates its segment with strong overall competitive metrics",
    "Challenger": "Strong performer with potential to overtake leaders",
    "Niche Player": "Focused player serving a specialized market segment",
}
 
# Reference date for computing company age (matches notebook)
REFERENCE_DATE = pd.Timestamp("2014-01-01")
 
 
# ---------------------------------------------------------------------------
# Step 1: Load & Clean
# ---------------------------------------------------------------------------
 
def load_companies(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load the startup_objects CSV and return only Company rows, cleaned."""
    companies = pd.read_csv(csv_path, usecols=LOAD_COLS, low_memory=False)
 
    # Keep only Company rows
    companies = companies[companies["entity_type"] == "Company"].copy()
 
    # Numeric columns: coerce errors → fill NaN with 0
    for col in NUMERIC_COLS:
        companies[col] = pd.to_numeric(companies[col], errors="coerce").fillna(0)
 
    # Category code: fill missing
    companies["category_code"] = companies["category_code"].fillna("unknown")
 
    # Parse founding date
    companies["founded_at"] = pd.to_datetime(companies["founded_at"], errors="coerce")
 
    print(f"[clustering] Loaded {len(companies):,} companies.")
    return companies
 
 
# ---------------------------------------------------------------------------
# Step 2: Feature Engineering
# ---------------------------------------------------------------------------
 
def add_company_age(companies: pd.DataFrame) -> pd.DataFrame:
    """Add 'company_age_years' column (age relative to REFERENCE_DATE)."""
    companies = companies.copy()
    age_days = (REFERENCE_DATE - companies["founded_at"]).dt.days
    companies["company_age_years"] = (
        (age_days / 365.25).clip(lower=0).fillna(0)
    )
    return companies
 
 
# ---------------------------------------------------------------------------
# Step 3: Random Forest — Strength Score
# ---------------------------------------------------------------------------
 
def train_strength_model(companies: pd.DataFrame) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor that predicts funding_total_usd from
    funding_rounds, milestones, relationships, and company_age_years.
    The predicted value is used as a proxy for 'company strength'.
    """
    X = companies[RF_TRAIN_FEATURES]
    y = companies["funding_total_usd"]
 
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
 
    importances = pd.Series(
        rf.feature_importances_, index=RF_TRAIN_FEATURES
    ).sort_values(ascending=False)
    print("[clustering] Feature importances (strength drivers):")
    print(importances.to_string())
 
    return rf
 
 
def add_strength_score(
    companies: pd.DataFrame, rf_model: RandomForestRegressor
) -> pd.DataFrame:
    """Append 'strength_score' column using the fitted RF model."""
    companies = companies.copy()
    companies["strength_score"] = rf_model.predict(companies[RF_TRAIN_FEATURES])
    return companies
 
 
# ---------------------------------------------------------------------------
# Step 4: K-Means Clustering — Market Position
# ---------------------------------------------------------------------------
 
def cluster_companies(companies: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Scale CLUSTER_FEATURES and fit K-Means (3 clusters).
    Returns the dataframe with 'cluster' and 'market_position' columns,
    plus the fitted KMeans and StandardScaler objects.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(companies[CLUSTER_FEATURES])
 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    companies = companies.copy()
    companies["cluster"] = kmeans.fit_predict(X_scaled)
 
    # Map cluster ids → human-readable position labels
    # (rank clusters by mean strength_score, highest = Market Leader)
    cluster_rank = (
        companies.groupby("cluster")["strength_score"]
        .mean()
        .sort_values(ascending=False)
    )
    label_map = {
        cluster_rank.index[0]: "Market Leader",
        cluster_rank.index[1]: "Challenger",
        cluster_rank.index[2]: "Niche Player",
    }
    companies["market_position"] = companies["cluster"].map(label_map)
 
    dist = companies["market_position"].value_counts()
    print("[clustering] Market position distribution:")
    print(dist.to_string())
 
    return companies, kmeans, scaler
 
 
# ---------------------------------------------------------------------------
# Step 5: Per-company SWOT — Strengths & Weaknesses
# ---------------------------------------------------------------------------
 
def generate_business_insight(row: pd.Series, companies: pd.DataFrame) -> dict:
    """
    Derive strengths and weaknesses for a single company row by
    normalising its metrics against its category peers.
    """
    strengths: list[str] = []
    weaknesses: list[str] = []
 
    category = row["category_code"]
    subset = companies[companies["category_code"] == category]
 
    def normalize(col: str) -> float:
        max_val = subset[col].max()
        min_val = subset[col].min()
        if max_val == min_val:
            return 0.5
        return (row[col] - min_val) / (max_val - min_val)
 
    def evaluate(metric_score: float, label: str) -> None:
        if metric_score >= 0.75:
            strengths.append(f"Strong {label} compared to competitors")
        elif metric_score <= 0.35:
            weaknesses.append(f"Weak {label} compared to top competitors")
        else:
            strengths.append(f"Average {label} performance")
 
    evaluate(normalize("funding_total_usd"), "financial strength")
    evaluate(normalize("funding_rounds"), "funding activity")
    evaluate(normalize("relationships"), "business network")
 
    return {
        "market_position": row["market_position"],
        "position_explanation": POSITION_EXPLANATIONS[row["market_position"]],
        "strengths": strengths,
        "weaknesses": weaknesses,
    }
 
 
# ---------------------------------------------------------------------------
# Pipeline entry-point: build the full enriched DataFrame
# ---------------------------------------------------------------------------
 
def build_competitor_dataframe(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Full pipeline:
      load → clean → feature engineer → RF strength score → K-Means cluster
 
    Returns the enriched companies DataFrame (used by competitor_agent.py).
    """
    companies = load_companies(csv_path)
    companies = add_company_age(companies)
 
    rf_model = train_strength_model(companies)
    companies = add_strength_score(companies, rf_model)
 
    companies, _kmeans, _scaler = cluster_companies(companies)
 
    return companies
 
 
# ---------------------------------------------------------------------------
# Keyword → category mapping (shared with competitor_agent.py)
# ---------------------------------------------------------------------------
 
KEYWORD_TO_CATEGORY: dict[str, str] = {
    "food": "hospitality",
    "delivery": "hospitality",
    "restaurant": "hospitality",
    "finance": "finance",
    "fintech": "finance",
    "health": "health",
    "fitness": "health",
    "education": "education",
    "learning": "education",
    "software": "software",
    "app": "mobile",
    "mobile": "mobile",
    "game": "games_video",
    "travel": "travel",
    "shopping": "ecommerce",
    "ecommerce": "ecommerce",
}
 
 
def match_category(keyword: str, all_categories: list[str]) -> str:
    """Map a free-text keyword to the closest Crunchbase category_code."""
    keyword = keyword.lower()
    for key, cat in KEYWORD_TO_CATEGORY.items():
        if key in keyword and cat in all_categories:
            return cat
    return "software"  # safe default