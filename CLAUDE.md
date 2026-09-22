# Orbital Sentinel

ML-powered satellite conjunction risk assessment system. Predicts collision probabilities from Conjunction Data Messages (CDMs) using the ESA Kelvins dataset (162K real CDMs, 102 features).

## Architecture

- `src/orbital_sentinel/` — Main package
  - `api/` — FastAPI REST API (port 8000)
  - `cli/` — Click CLI (`orbital-sentinel` command)
  - `config/` — Settings and YAML configuration
  - `database/` — SQLAlchemy persistence (SQLite/PostgreSQL)
  - `evaluation/` — Model evaluation, metrics, calibration
  - `events/` — Event tracking, CDM sequencing, reconstruction
  - `explainability/` — SHAP analysis and natural language explanations
  - `features/` — Feature engineering (static, orbital, covariance, temporal)
  - `fusion/` — Risk fusion (ML + physics, confidence, disagreement)
  - `inference/` — Prediction pipeline (single event + batch)
  - `ingestion/` — Data loading, Space-Track CDM API client
  - `models/` — ML models (XGBoost, LightGBM, RF, Logistic, LSTM, GRU, Transformer)
  - `monitoring/` — Data quality monitoring, drift detection
  - `physics/` — Collision probability, covariance validation, Monte Carlo
  - `preprocessing/` — Cleaning, normalization, event-based splitting
  - `reporting/` — HTML report generation (conjunction, model, timeline)
  - `alerts/` — Alert rules, manager, webhook/log notifications
  - `uncertainty/` — Prediction intervals, calibration
  - `validation/` — Schema validation, quality checks, leakage detection
  - `visualization/` — 3D orbit visualization (Plotly)
- `dashboard/` — Streamlit dashboard (9 pages, dark glassmorphism theme)
- `scripts/` — Pipeline runner
- `tests/` — Test suite (pytest)
- `configs/config.yaml` — Project configuration
- `data/raw/esa_kelvins/` — Training data (not in git)
- `models_saved/` — Trained model artifacts

## Key Commands

```bash
pip install -e ".[dev]"          # Install with dev dependencies
python scripts/run_pipeline.py   # Train models (11-step pipeline)
python -m orbital_sentinel.api   # Start API server (port 8000)
streamlit run dashboard/app.py   # Start dashboard (port 8501)
orbital-sentinel --help          # CLI tool
pytest tests/ -v                 # Run tests
docker-compose up -d             # Start full stack
```

## Risk Thresholds

- HIGH: log10(Pc) > -5
- MEDIUM: -7 to -5
- LOW: -15 to -7
- NEGLIGIBLE: < -15

## Target Column

`risk` = log10(collision probability). Floor value: -30.0.
