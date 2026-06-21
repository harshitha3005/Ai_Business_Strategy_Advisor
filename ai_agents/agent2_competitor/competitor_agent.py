"""
competitor_agent.py — Agent 2: Competitor Intelligence AI Agent
 
Part of the AI Business Strategy Advisor multi-agent system.
This module is the production entry-point for Agent 2. It:
  - Loads the enriched companies DataFrame once at startup (lazy singleton)
  - Exposes agent_competitor(keyword) which returns a structured JSON-ready dict
  - Can be invoked directly (CLI) or imported by the FastAPI backend planner
 
File location (per project structure):
  ai_agents/agent2_competitor/competitor_agent.py
 
Depends on:
  ai_agents/agent2_competitor/clustering.py  ← ML pipeline
  data/startup_objects.csv                   ← Crunchbase dataset
 
Output schema (returned by agent_competitor):
  {
    "agent": "Competitor Intelligence Agent",
    "input_keyword": <str>,
    "matched_category": <str>,
    "num_companies_analyzed": <int>,
    "competitors": [
      {
        "company_name": <str>,
        "country": <str>,
        "market_position": "Market Leader" | "Challenger" | "Niche Player",
        "position_explanation": <str>,
        "strengths": [<str>, ...],
        "weaknesses": [<str>, ...]
      },
      ...  # top_n results (default 5)
    ]
  }
"""
 
from __future__ import annotations
 
import json
import os
import sys
from typing import Any
 
# ---------------------------------------------------------------------------
# Imports from clustering module (same package)
# ---------------------------------------------------------------------------
 
# Allow running from repo root as well as from inside agent2_competitor/
sys.path.insert(0, os.path.dirname(__file__))
 
from clustering import (
    build_competitor_dataframe,
    generate_business_insight,
    match_category,
)
 
import pandas as pd
 
# ---------------------------------------------------------------------------
# Lazy singleton — the enriched DataFrame is expensive to build (RF training),
# so we build it once and cache it in the module-level variable.
# ---------------------------------------------------------------------------
 
_companies: pd.DataFrame | None = None
 
 
def _get_companies() -> pd.DataFrame:
    """Return the cached enriched DataFrame, building it on first call."""
    global _companies
    if _companies is None:
        print("[competitor_agent] Building competitor database … (first call only)")
        _companies = build_competitor_dataframe()
        print("[competitor_agent] Database ready.")
    return _companies
 
 
# ---------------------------------------------------------------------------
# Core agent function
# ---------------------------------------------------------------------------
 
def agent_competitor(keyword: str, top_n: int = 5) -> dict[str, Any]:
    """
    Competitor Intelligence Agent entry-point.
 
    Args:
        keyword: Free-text business idea or industry keyword
                 e.g. "food delivery", "fintech", "health app"
        top_n:   Number of top competitors to return (default 5)
 
    Returns:
        Structured dict matching the Agent 2 output schema (JSON-serialisable).
    """
    companies = _get_companies()
 
    all_categories = companies["category_code"].unique().tolist()
    category = match_category(keyword, all_categories)
 
    # Filter to the matched category
    subset = companies[companies["category_code"] == category]
 
    if subset.empty:
        return {
            "agent": "Competitor Intelligence Agent",
            "input_keyword": keyword,
            "matched_category": category,
            "num_companies_analyzed": 0,
            "competitors": [],
            "warning": f"No companies found for category '{category}'. "
                       "Try a different keyword.",
        }
 
    # Pick top_n companies by strength_score
    top_companies = (
        subset.sort_values("strength_score", ascending=False).head(top_n)
    )
 
    competitors: list[dict[str, Any]] = []
    for _, row in top_companies.iterrows():
        insight = generate_business_insight(row, companies)
        competitors.append(
            {
                "company_name": row["name"],
                "country": row.get("country_code", "N/A"),
                "market_position": insight["market_position"],
                "position_explanation": insight["position_explanation"],
                "strengths": insight["strengths"],
                "weaknesses": insight["weaknesses"],
            }
        )
 
    return {
        "agent": "Competitor Intelligence Agent",
        "input_keyword": keyword,
        "matched_category": category,
        "num_companies_analyzed": int(len(subset)),
        "competitors": competitors,
    }
 
 
# ---------------------------------------------------------------------------
# FastAPI-compatible async wrapper
# (used by backend/app/services/planner.py when calling agents in parallel)
# ---------------------------------------------------------------------------
 
async def run_agent(keyword: str, top_n: int = 5) -> dict[str, Any]:
    """
    Async wrapper around agent_competitor for use with FastAPI / asyncio.
    The ML inference is CPU-bound but fast after first load, so we run
    it synchronously inside the async context (acceptable for college-scale
    load; for production consider running in a ThreadPoolExecutor).
    """
    return agent_competitor(keyword, top_n=top_n)
 
 
# ---------------------------------------------------------------------------
# CLI entry-point — run directly: python competitor_agent.py "food delivery"
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(
        description="Agent 2 — Competitor Intelligence AI Agent"
    )
    parser.add_argument(
        "keyword",
        nargs="?",
        default="food delivery",
        help="Business idea keyword (default: 'food delivery')",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top competitors to return (default: 5)",
    )
    args = parser.parse_args()
 
    result = agent_competitor(args.keyword, top_n=args.top_n)
    print(json.dumps(result, indent=2, default=str))
 