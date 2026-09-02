# Financial-Crime-AI

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

An **AI-Powered Financial Crime Investigation Platform** for AML/Fraud
analysts. The system ingests transactions, detects anomalies with multiple
ML models, scores risk, ranks cases into an investigation queue and drives
an **AI investigator agent** (RAG + structured tool calling) that explains
*why* each transaction was flagged and recommends the next action — all
surfaced through an analyst dashboard.

```
                   Transaction data
                          │
                          ▼
                   Data pipeline
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
      Transaction ML               Graph features
      anomaly model                     │
            │                           │
            └─────────────┬─────────────┘
                          ▼
                   Risk scoring model
                          │
                          ▼
                   Investigation queue
                          │
                          ▼
                 ┌──────────────────┐
                 │  AI Investigator │
                 │      Agent       │
                 └────────┬─────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
         Evidence      Explain       Recommend
         retrieval     anomaly        action
                          │
                          ▼
                    Analyst dashboard
```

An analyst clicks a case like **Transaction #T00194810 — risk 0.99** and
sees: why it was flagged, matched AML typologies, connected accounts,
historical behaviour, similar cases, supporting evidence, model confidence
and a recommended action.

---

## Architecture

| Stage | Kedro pipeline | What it does |
|---|---|---|
| Data ingestion | `data_ingestion` | Deterministic synthetic generation of customers, accounts, a sanctions/watchlist and ~200k transactions with six injected money-laundering patterns (structuring, layering/mule chains, rapid cash-out, gambling churn, round-amount bulk, sanctions flows). Ground-truth labels are produced for evaluation. |
| Feature engineering | `feature_engineering` | Transaction features (velocity over 24h/7d, amount z-score vs the account's own history, threshold proximity, channel/country risk, time gaps) + **graph features** over a directed money-flow graph: degree/fan-in/fan-out, PageRank, HITS hub score, k-core, community size, shortest path to a sanctioned entity, structuring & layering evidence, and 8-d SVD **entity embeddings**. |
| Anomaly detection | `anomaly_detection` | **Baseline**: Isolation Forest (unsupervised). **Supervised**: XGBoost classifier. Both evaluated with a temporal holdout. |
| Risk scoring | `risk_scoring` | XGBoost risk model combining anomaly outputs + graph features into a calibrated per-transaction risk probability; per-account aggregation. |
| Investigation | `investigation` | Flags transactions above a threshold, de-duplicates into one case per account, ranks by risk and snapshots structured evidence + a similar-case index. |
| AI investigator | `agent` | RAG evidence retrieval, SHAP-style feature attribution, AML typology matching, similar-case retrieval, sanctions checks and recommended actions. **Structured tool calling** with a real LLM when `OPENAI_API_KEY` is set; a deterministic engine otherwise. |

## ML roadmap coverage

**Start (baseline):**
- Isolation Forest unsupervised anomaly score
- XGBoost supervised anomaly detection

**Advanced:**
- Temporal anomaly features (rolling 24h / 7d velocity and value)
- Graph features (centrality, community, k-hop sanctions reach, flow shares)
- **Entity embeddings** (SVD structural embeddings, a drop-in point for a full GNN)
- GNN (future extension point — see `graph_features._entity_embeddings`)

**GenAI:**
- RAG investigator assistant (typology knowledge base + similar-case retrieval)
- Structured tool calling (LLM agent loop over a tool registry)
- Case summarisation (LLM narrative or deterministic template)

## Model quality

Risk model evaluated on a **temporal holdout** (train on the first 80% of the
60-day window, score the most recent 20% — mirroring real AML deployment):

```
  split      auc       ap      n
  train 0.999999 0.999997 165536
holdout 0.999552 0.995054  41385
```

## Getting started

```bash
# 1. Create a virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

# 2. Run the full pipeline (generates data, trains models, builds queue)
.venv/bin/kedro run

# 3. Launch the analyst dashboard
./scripts/run_dashboard.sh            # streamlit

# 4. Investigate a single case from the CLI
./scripts/run_investigation.sh T00194810
```

Individual stages can be re-run independently:
`kedro run --pipeline=features`, `--pipeline=anomaly`, `--pipeline=risk`,
`--pipeline=investigation`.

## Enabling the LLM agent

The AI investigator runs fully offline via a deterministic engine by
default. To use the **LLM agent with structured tool calling**, set:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini        # optional
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
```

The dashboard shows which engine generated each report
(`llm-agent` vs `deterministic-engine`).

## Tests & linting

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

## Project layout

```
conf/base/            catalog + parameters
src/financial_crime_ai/
  features.py         shared feature definitions + AML typology knowledge base
  pipelines/          Kedro pipelines (data_ingestion → agent)
  agent/              AI investigator (tools, RAG, explainer, LLM, fallback)
  dashboard/          Streamlit analyst dashboard
data/                 pipeline outputs (generated, git-ignored)
```

## Rules and guidelines

* Don't remove lines from the `.gitignore` file we provide.
* Results are reproducible: everything is seeded and the data pipeline is deterministic.
* Don't commit data or credentials — keep them in `conf/local/` / `data/`.
