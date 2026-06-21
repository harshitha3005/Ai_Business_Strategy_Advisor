# AI/ML Engineer (Agents 1–4) — File Structure & Notebooks

This folder corresponds to **your role** in the *AI Business Strategy Advisor* project
(Team Member 3 — AI/ML Engineer, Agents 1–4), based on the "8. File Structure by Team
Member" section of the project documentation.

```
ai_agents/
├── agent1_market/
│   ├── market_agent.py        ← (to build) wraps agent_market() for FastAPI
│   ├── forecasting.py         ← (to build) Prophet/ARIMA helper functions
│   └── notebooks/
│       └── market_analysis.ipynb     ✅ created
├── agent2_competitor/
│   ├── competitor_agent.py    ← (to build) wraps agent_competitor()
│   ├── clustering.py          ← (to build) K-Means + Random Forest helpers
│   └── notebooks/
│       └── competitor_analysis.ipynb ✅ created
├── agent3_customer/
│   ├── customer_agent.py      ← (to build) wraps agent_customer()
│   ├── sentiment.py           ← (to build) BERT sentiment + TF-IDF helpers
│   └── notebooks/
│       └── sentiment_analysis.ipynb  ✅ created
├── agent4_social/
│   ├── social_agent.py        ← (to build) wraps agent_social()
│   ├── topic_modeling.py       ← (to build) LDA helpers
│   └── notebooks/
│       └── trend_analysis.ipynb      ✅ created
└── data/                        ✅ datasets copied & renamed here
    ├── startup_objects.csv          (from "startup investment/objects.csv")
    ├── startup_funding_rounds.csv   (from "startup investment/funding_rounds.csv")
    ├── worldbank_indicators.csv     (from "world_bank_indicators.csv")
    ├── twitter_sentiment.csv        (from "Twitter sentiment dataset/Tweets.csv")
    └── amazon_reviews_sample.csv    (from "Amazon consumer review/1429_1.csv")
```

## What each notebook does

### 1. `agent1_market/notebooks/market_analysis.ipynb` — Market Research Agent
- Loads Crunchbase company data (`startup_objects.csv`) and funding rounds
  (`startup_funding_rounds.csv`).
- Builds a category × year funding table and computes growth % + a next-year
  forecast (simple linear trend — swap for Prophet/ARIMA if you have those
  packages).
- Pulls global GDP growth from `worldbank_indicators.csv`.
- Exposes `agent_market(keyword)` → industry growth %, funding trend,
  recommended sector (matches docs section "Agent 1").

### 2. `agent2_competitor/notebooks/competitor_analysis.ipynb` — Competitor Intelligence Agent
- Uses the same Crunchbase company data.
- Engineers features (funding, rounds, milestones, relationships, age).
- Trains a **Random Forest** to score each company's "strength" and reports
  feature importances.
- Runs **K-Means** (3 clusters) and labels them Leader / Challenger / Niche
  by average funding.
- Exposes `agent_competitor(keyword)` → ranked competitor table with
  cluster labels and strength scores.

### 3. `agent3_customer/notebooks/sentiment_analysis.ipynb` — Customer Insight Agent
- Loads `amazon_reviews_sample.csv`.
- Runs HuggingFace **BERT** sentiment analysis (falls back to a
  rating-based label if the model can't be downloaded offline).
- Uses **TF-IDF** to pull top complaint/praise keywords from negative vs.
  positive reviews.
- Exposes `agent_customer(category)` → top positives, top negatives,
  sentiment distribution, summary insight.

### 4. `agent4_social/notebooks/trend_analysis.ipynb` — Social Media Trend Agent
- Loads `twitter_sentiment.csv` (Twitter Airline Sentiment dataset).
- Cleans tweets (mentions/links/stopwords removed).
- Runs **LDA topic modeling** (scikit-learn) to find trending themes.
- Computes a sentiment-based "public interest score" per query.
- Exposes `agent_social(keyword)` → trending topics, interest score,
  sentiment distribution, recommendation.

## How to run

1. Open each notebook in Jupyter (the relative path `../../data/...` already
   points at the `data/` folder containing the renamed datasets).
2. Run all cells top-to-bottom — each notebook ends with a working
   `agent_xxx(keyword)` function and a test call.
3. (Optional, recommended for Step "Wrap each agent as a Python function")
   move the final function + its dependencies out of the notebook into the
   matching `*_agent.py` / helper `.py` file listed above, so the FastAPI
   backend (Member 2) can `import` and call them directly.

## Notes / caveats
- `startup_objects.csv` is large (~280MB); the notebooks use `usecols` to
  only load the needed columns so it stays manageable.
- BERT sentiment in notebook 3 needs internet access the first time to
  download the model — it degrades gracefully to rating-based sentiment
  otherwise.
- Category matching (`KEYWORD_TO_CATEGORY`) is a simple keyword map against
  real Crunchbase `category_code` values (e.g. food → `hospitality`,
  fintech → `finance`). Extend this dictionary as needed.
