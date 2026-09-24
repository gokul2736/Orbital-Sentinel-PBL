# Orbital Sentinel — Technical Documentation

**ML-Powered Satellite Conjunction Risk Assessment System**

Version 2.0 | September 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Machine Learning Pipeline](#3-machine-learning-pipeline)
4. [Physics Engine](#4-physics-engine)
5. [REST API](#5-rest-api)
6. [Command-Line Interface](#6-command-line-interface)
7. [Dashboard](#7-dashboard)
8. [Database Layer](#8-database-layer)
9. [Monitoring and Alerts](#9-monitoring-and-alerts)
10. [Testing Strategy](#10-testing-strategy)
11. [Deployment](#11-deployment)
12. [Complete Feature List](#12-complete-feature-list)

---

## 1. Project Overview

### Problem Statement

Space debris poses a growing threat to operational satellites and crewed spacecraft. Over 36,000 objects larger than 10 cm are tracked in low-Earth orbit, generating thousands of conjunction alerts daily. Satellite operators must rapidly assess collision risk from Conjunction Data Messages (CDMs) to decide whether evasive maneuvers are warranted — decisions that carry significant cost and operational impact.

**Orbital Sentinel** automates this triage process using machine learning to predict collision probability from CDM features, validated against first-principles physics computations. The system fuses ML predictions with analytic collision probability estimates to produce calibrated risk assessments with uncertainty quantification and natural-language explanations.

### Dataset

The system is trained on the **ESA Kelvins Conjunction Dataset**, a benchmark dataset released by the European Space Agency containing:

- **162,000+ real Conjunction Data Messages** from operational conjunction assessment
- **102 raw features** per CDM, including orbital elements (J2000), covariance matrices (RTN frame), miss distance, relative velocity, time-to-TCA, and object classification
- **Target variable**: `risk` = log10(collision probability), with a floor value of −30.0
- **Event structure**: CDMs are grouped by `event_id`, with multiple CDMs per conjunction event representing updates as TCA approaches

### Risk Classification

| Category    | log10(Pc) Range | Interpretation                          |
|-------------|----------------|-----------------------------------------|
| HIGH        | > −5           | Potential maneuver candidate             |
| MEDIUM      | −7 to −5       | Elevated risk, increased monitoring      |
| LOW         | −15 to −7      | Standard tracking, no action required    |
| NEGLIGIBLE  | < −15          | Effectively zero risk                    |

---

## 2. System Architecture

### High-Level Data Flow

```
              CONJUNCTION DATA
                     │
         ┌───────────┴───────────┐
         ↓                       ↓
   ESA KELVINS              SPACE-TRACK
   Historical                Live CDM API
         │                       │
         └───────────┬───────────┘
                     ↓
              DATA INGESTION
                     ↓
             VALIDATION / CLEANING
                     ↓
           EVENT RECONSTRUCTION
                     ↓
            FEATURE ENGINEERING
                     ↓
              144 FEATURES
                     ↓
          EVENT-BASED SPLITTING
                     ↓
      ┌──────────────┼──────────────┐
      ↓              ↓              ↓
   XGBoost       Temporal        Physics
   LightGBM      Models          Engine
   RF / Ridge    (LSTM/GRU/      (Analytic Pc,
                  Transformer)    Monte Carlo)
      ↓              ↓              ↓
      └──────────────┼──────────────┘
                     ↓
              STACKING ENSEMBLE
                     ↓
                RISK FUSION
                     ↓
              UNCERTAINTY QUANTIFICATION
                     ↓
             SHAP EXPLAINABILITY
                     ↓
              DECISION SUPPORT
                     ↓
     ┌──────────┬──────────┬──────────┐
     ↓          ↓          ↓          ↓
  REST API   CLI Tool   Dashboard   Alerts
  (FastAPI)  (Click)    (Streamlit) (Webhook)
```

### Component Overview

| Component           | Technology       | Purpose                                    |
|---------------------|------------------|--------------------------------------------|
| ML Models           | scikit-learn, XGBoost, LightGBM, PyTorch | Risk prediction       |
| Physics Engine      | NumPy, SciPy    | Analytic collision probability              |
| Feature Engineering | Pandas, NumPy   | 144 derived features from 102 raw columns   |
| REST API            | FastAPI          | Programmatic access (port 8000)             |
| CLI                 | Click            | Command-line pipeline operations            |
| Dashboard           | Streamlit, Plotly | Interactive visualization (port 8501)      |
| Database            | SQLAlchemy       | Persistent storage (SQLite/PostgreSQL)      |
| Monitoring          | SciPy (KS test) | Data drift detection                        |
| Alerts              | Custom rules     | Configurable alert notifications            |
| CI/CD               | GitHub Actions   | Automated lint, test, type-check, Docker    |
| Containerization    | Docker Compose   | Multi-service deployment                    |

### Package Structure

```
src/orbital_sentinel/
├── api/                    # FastAPI REST API
│   ├── app.py              # Application factory
│   ├── dependencies.py     # Dependency injection
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/
│       ├── predictions.py  # Single, quick, batch prediction endpoints
│       ├── cdm.py          # CDM fetch, analysis, sample data
│       ├── health.py       # Health check, system status
│       └── models.py       # Model listing, info, features
├── cli/                    # Click CLI
│   ├── main.py             # CLI entry point
│   ├── formatting.py       # Output formatting
│   └── commands/
│       ├── train.py        # Model training command
│       ├── predict.py      # Prediction command
│       ├── serve.py        # API server launcher
│       ├── data.py         # Data management
│       └── status.py       # System status
├── config/                 # Configuration management
│   └── settings.py         # YAML-based settings
├── database/               # Persistence layer
│   ├── connection.py       # Engine and session management
│   ├── models.py           # ORM models (6 tables)
│   ├── migrations.py       # Schema versioning
│   └── repository.py       # Data access patterns
├── evaluation/             # Model evaluation
│   ├── metrics.py          # RMSE, MAE, R², correlation
│   ├── calibration.py      # Probability calibration
│   ├── cross_validation.py # K-fold cross-validation
│   ├── reports.py          # Evaluation report generation
│   ├── thresholding.py     # Risk threshold analysis
│   └── threshold_optimization.py
├── events/                 # Conjunction event management
│   ├── reconstruction.py   # Event reconstruction from CDMs
│   ├── sequencing.py       # CDM sequence analysis, risk trends
│   └── tracking.py         # Event lifecycle tracking
├── explainability/         # Model interpretability
│   ├── shap_analysis.py    # SHAP value computation
│   └── explanations.py     # Natural language explanations
├── features/               # Feature engineering
│   ├── static.py           # Static CDM features
│   ├── orbital.py          # Orbital mechanics features
│   ├── covariance.py       # Covariance-derived features
│   └── temporal.py         # Time-series features
├── fusion/                 # Risk fusion engine
│   ├── risk.py             # ML + physics fusion
│   ├── confidence.py       # Confidence scoring
│   └── disagreement.py     # ML-physics disagreement analysis
├── inference/              # Production inference
│   └── pipeline.py         # Single event and batch prediction
├── ingestion/              # Data loading
│   ├── dataset_loader.py   # ESA Kelvins CSV loader
│   ├── cdm_api.py          # Space-Track.org API client
│   ├── cdm_mapper.py       # Raw CDM to ESA format mapper
│   └── api_client.py       # HTTP client utilities
├── models/                 # ML model implementations
│   ├── ensemble.py         # Stacking ensemble (RidgeCV meta-learner)
│   ├── selection.py        # Model selection utilities
│   ├── tuning.py           # Optuna hyperparameter optimization
│   ├── baselines/
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── random_forest.py
│   │   └── logistic.py     # Ridge regression
│   └── temporal/
│       ├── lstm.py          # Long Short-Term Memory
│       ├── gru.py           # Gated Recurrent Unit
│       └── transformer.py   # Transformer encoder
├── monitoring/             # Production monitoring
│   ├── data_quality.py     # Data quality checks
│   └── drift.py            # Distribution drift detection (KS, PSI)
├── physics/                # First-principles physics
│   ├── collision_probability.py  # 2D short-encounter Pc
│   ├── covariance.py       # Covariance operations
│   ├── geometry.py         # Conjunction geometry (RTN frame)
│   ├── monte_carlo.py      # Monte Carlo Pc estimation
│   └── propagation.py      # Orbit propagation
├── preprocessing/          # Data preparation
│   ├── cleaning.py         # Missing values, outliers
│   ├── normalization.py    # Feature scaling
│   └── splitting.py        # Event-based train/test split
├── reporting/              # Report generation
│   ├── conjunction_report.py    # Per-event HTML reports
│   ├── model_report.py         # Model evaluation reports
│   ├── event_timeline_report.py # Timeline visualizations
│   └── templates.py        # HTML report templates
├── alerts/                 # Alert system
│   ├── rules.py            # Alert rule definitions
│   ├── manager.py          # Alert evaluation engine
│   └── notifier.py         # Webhook and log notifications
├── uncertainty/            # Uncertainty quantification
│   ├── predictive.py       # Prediction intervals
│   └── calibration.py      # Interval calibration
├── validation/             # Data validation
│   ├── schema.py           # Schema validation
│   ├── quality.py          # Quality checks
│   └── leakage.py          # Data leakage detection
└── visualization/          # 3D visualization
    └── orbit_3d.py         # Plotly orbit rendering

dashboard/
├── app.py                  # Main dashboard (14 pages)
└── _pages/
    ├── orbital_tracker.py  # Animated conjunction visualization
    ├── ensemble_comparison.py  # Model comparison analytics
    ├── alert_dashboard.py  # Alert monitoring
    ├── report_generator.py # HTML report generation
    └── system_status.py    # Infrastructure health

scripts/
├── run_pipeline.py         # 11-step training pipeline
├── test_live_cdm.py        # Live CDM fetch testing
└── test_space_track.py     # Space-Track API testing

tests/                      # 16 test modules
├── test_models.py
├── test_physics.py
├── test_features.py
├── test_preprocessing.py
├── test_evaluation.py
├── test_fusion.py
├── test_uncertainty.py
├── test_explainability.py
├── test_validation.py
├── test_ingestion.py
├── test_api.py
├── test_database.py
├── test_alerts.py
├── test_reporting.py
├── test_cdm_api.py
└── test_ml_advanced.py
```

---

## 3. Machine Learning Pipeline

### 3.1 Training Pipeline

The 11-step pipeline (`scripts/run_pipeline.py`) executes the full workflow:

1. **Load Dataset** — Read ESA Kelvins CSV (162K rows, 102 columns)
2. **Validate Schema** — Verify required columns, data types, ranges
3. **Quality Checks** — Detect outliers, negative values, missing data patterns
4. **Clean Data** — Handle missing values, remove invalid rows, clip outliers
5. **Feature Engineering** — Generate 144 features from 102 raw columns
6. **Event-Based Splitting** — Split train/test by event ID (prevents data leakage)
7. **Train Baseline Models** — XGBoost, LightGBM, Random Forest, Ridge
8. **Evaluate Models** — RMSE, MAE, R², correlation per model and per risk band
9. **Physics Verification** — Run analytic Pc on test set sample
10. **Risk Fusion** — Combine ML predictions with physics results
11. **Save Artifacts** — Model files, scaler, feature names, pipeline results

### 3.2 Models

#### Baseline Models (scikit-learn compatible)

| Model         | Library   | Key Parameters                          |
|---------------|-----------|----------------------------------------|
| XGBoost       | xgboost   | 200–1000 trees, depth 4–12, early stopping |
| LightGBM      | lightgbm  | 200–1000 trees, 20–127 leaves          |
| Random Forest | sklearn   | 100–600 trees, depth 6–20              |
| Ridge         | sklearn   | L2 regularization (alpha cross-validated) |

#### Temporal Models (PyTorch)

| Model       | Architecture                          | Purpose                           |
|-------------|---------------------------------------|-----------------------------------|
| LSTM        | 2-layer bidirectional LSTM + FC       | CDM sequence risk evolution       |
| GRU         | 2-layer GRU + FC                      | Lightweight temporal modeling     |
| Transformer | Multi-head self-attention encoder + FC | Long-range CDM dependencies       |

#### Stacking Ensemble

The `StackingEnsemble` combines all base model predictions using a **RidgeCV meta-learner**:

1. Each base model produces a prediction on the training data
2. Base predictions become meta-features (one column per model)
3. RidgeCV learns optimal combination weights with cross-validated regularization (alpha ∈ {0.01, 0.1, 1.0, 10.0})
4. Final prediction = meta-learner output

### 3.3 Hyperparameter Tuning

**Optuna** (Tree-structured Parzen Estimator) tunes each model's hyperparameters:

- **XGBoost**: 50 trials over 8 parameters (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda)
- **LightGBM**: 50 trials over 9 parameters (adds num_leaves, min_child_samples)
- **Random Forest**: 30 trials over 5 parameters (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
- **Objective**: Minimize validation RMSE

### 3.4 Feature Engineering

144 features are derived from the 102 raw CDM columns across four categories:

#### Static Features
- Raw CDM fields: miss distance, relative speed, mahalanobis distance
- Object classification: target/chaser type encoding
- Cross-section ratios, area-to-mass computations

#### Orbital Features
- Semi-major axis, eccentricity, inclination (J2000 frame)
- Orbital period, mean motion derivatives
- Perigee/apogee altitude ratios
- Relative orbital energy and angular momentum

#### Covariance Features
- Covariance matrix eigenvalues (target and chaser)
- Covariance trace, determinant, condition number
- Cross-correlation between target and chaser uncertainty
- Position uncertainty volume

#### Temporal Features
- Time-to-TCA (days)
- CDM update frequency within event
- Risk change rate between CDMs
- Convergence indicators

### 3.5 Event-Based Splitting

Standard random splitting would leak information between CDMs from the same conjunction event. The system uses **event-based splitting** where all CDMs belonging to the same event appear exclusively in either the training or test set:

```
Split strategy: group-by event_id
Train: 80% of events (all their CDMs)
Test:  20% of events (all their CDMs)
Random state: 42 (reproducible)
```

### 3.6 Evaluation Metrics

**Regression Metrics** (primary task):
- RMSE (Root Mean Squared Error) — primary optimization metric
- MAE (Mean Absolute Error) — robust to outliers
- R² (Coefficient of Determination)
- Pearson Correlation

**Binary Classification Metrics** (at threshold −5.0):
- Precision, Recall, F1 Score
- AUC (Area Under ROC Curve)

**Per-Risk-Band Evaluation**: Separate metrics for HIGH, MEDIUM, LOW, NEGLIGIBLE bands to ensure balanced performance across the risk spectrum.

---

## 4. Physics Engine

### 4.1 Collision Probability (2D Short-Encounter)

Implements the **Foster (1992) / Akella-Alfriend** short-encounter Pc formula, the standard method used by NASA, ESA, and the 18th Space Defense Squadron:

**Method**: Numerical integration of a 2D Gaussian probability density over a circular hard-body disk:

```
Pc = ∫∫_disk(R) N(μ, C) dA
```

Where:
- μ = miss vector projected onto the encounter (B-plane)
- C = 2×2 combined covariance in the encounter plane
- R = combined hard-body radius (default 20 m)

**Implementation**: Alfano grid integration — evaluates the Gaussian PDF on a 200×200 grid over the hard-body disk and sums contributions within the circular boundary.

**Cross-check**: Foster analytic approximation (first-order, valid when R << σ):
```
Pc ≈ (R²) / (2·σ_x'·σ_y') · exp(−½ · d'^T C'^−1 d')
```

### 4.2 Covariance Validation

Before computing Pc, all covariance matrices undergo validation:

- **Symmetry check**: |C − C^T| < ε
- **Positive semi-definiteness**: All eigenvalues > 0
- **Condition number**: Flags ill-conditioned matrices
- **NaN/Inf detection**: Rejects non-finite values
- **Physical plausibility**: Ensures uncertainty magnitudes are realistic

### 4.3 Monte Carlo Verification

Independent verification of the analytic Pc using Monte Carlo sampling:

1. Draw N = 10,000 samples from the combined 3D Gaussian distribution
2. Count samples where |miss vector| < combined hard-body radius
3. MC Pc = hits / N
4. Compare with analytic Pc for consistency
5. Report agreement (within 1 order of magnitude tolerance)

### 4.4 Conjunction Geometry

Full geometric analysis in the RTN (Radial-Transverse-Normal) frame:

- **Relative state**: Position and velocity vectors at TCA
- **Miss distance decomposition**: R, T, N components
- **Approach angle**: Computed from relative velocity
- **Encounter type**: Head-on, crossing, co-orbital classification
- **Encounter duration**: Based on relative velocity and uncertainty size

### 4.5 Risk Fusion (ML + Physics)

The fusion engine combines ML predictions with physics verification using a weighted average with adaptive trust:

```
Fused Risk = (1 − w_physics) · ML_pred + w_physics · Physics_log10Pc
```

**Adaptive weight adjustment**:
- Default physics weight: 0.3
- **Severe disagreement** (|ML − Physics| > 5): physics weight increases to min(0.6, w + 0.2)
- **Significant disagreement**: physics weight increases to min(0.5, w + 0.1)
- **Invalid covariance**: physics weight halved (reduced trust)
- **Short-encounter violation**: physics weight reduced by 30%
- **No physics available**: ML-only with 25% wider prediction interval

**Confidence scoring** incorporates:
- Prediction interval width (narrow = high confidence)
- ML-physics agreement (agreement = higher confidence)
- Covariance quality (valid = higher confidence)
- Time to TCA (closer = higher confidence from better data)

---

## 5. REST API

### Base URL: `http://localhost:8000`

### Endpoints

| Method | Path                 | Description                              |
|--------|----------------------|------------------------------------------|
| GET    | `/health`            | Health check with model status and uptime |
| GET    | `/api/v1/status`     | Detailed system status                   |
| POST   | `/api/v1/predict`    | Full prediction (ML + physics + SHAP)    |
| POST   | `/api/v1/predict/quick` | Fast prediction (ML only)             |
| POST   | `/api/v1/predict/batch` | Batch prediction for multiple CDMs    |
| POST   | `/api/v1/cdm/fetch`  | Fetch live CDMs from Space-Track         |
| POST   | `/api/v1/cdm/analyze`| Full CDM analysis                        |
| GET    | `/api/v1/cdm/sample` | Sample from ESA Kelvins dataset          |
| GET    | `/api/v1/models`     | List available trained models            |
| GET    | `/api/v1/models/info`| Current model information                |
| GET    | `/api/v1/models/features` | Feature names list                  |
| GET    | `/api/v1/models/{name}` | Specific model details + metrics      |

### Request/Response Example

**POST `/api/v1/predict`**

Request body: CDM feature dictionary (miss_distance, relative_speed, orbital elements, covariance entries, etc.)

Response:
```json
{
  "prediction": -6.42,
  "risk_category": "MEDIUM",
  "confidence": 0.78,
  "interval": [-8.1, -4.8],
  "physics_result": {
    "log10_pc": -6.8,
    "covariance_valid": true,
    "short_encounter_valid": true
  },
  "fusion": {
    "fused_risk": -6.53,
    "risk_category": "MEDIUM",
    "ml_physics_agreement": "AGREE"
  },
  "explanation": "Risk is MEDIUM due to moderate miss distance...",
  "top_features": [
    {"feature": "miss_distance", "shap_value": -2.3},
    {"feature": "mahalanobis_distance", "shap_value": 1.8}
  ]
}
```

### Auto-generated Documentation

FastAPI provides interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 6. Command-Line Interface

### Installation

```bash
pip install -e ".[dev]"
orbital-sentinel --help
```

### Commands

| Command                        | Description                              |
|-------------------------------|------------------------------------------|
| `orbital-sentinel train`      | Run model training pipeline              |
| `orbital-sentinel predict`    | Run prediction on CDM input              |
| `orbital-sentinel serve`      | Start FastAPI server                     |
| `orbital-sentinel data`       | Data management (load, validate, stats)  |
| `orbital-sentinel status`     | Show system status and model info        |

---

## 7. Dashboard

### Overview

The Streamlit dashboard provides a 14-page interactive interface with a **dark glassmorphism design theme** featuring:

- Custom CSS with CSS variables for consistent theming
- Glassmorphism effects (frosted glass panels with backdrop-filter blur)
- JetBrains Mono monospace font for numeric data
- Animated pulse indicators for live system status
- Responsive layout with persistent sidebar navigation
- Plotly interactive charts with dark theme integration

### Pages

#### 1. Mission Control
Operational overview dashboard displaying:
- Hero metrics row (Training CDMs, Unique Events, High Risk, Best RMSE, Model R²)
- Live CDM quick-fetch from Space-Track with real-time event cards
- Risk distribution histogram with threshold markers
- Alert feed showing high-risk conjunction events
- Risk category breakdown (donut chart + statistics)
- Model leaderboard table (sortable by metric)

#### 2. Live CDM Feed
Real-time conjunction data from Space-Track.org:
- API status indicator and model load status
- Configurable CDM fetch (1–20 at a time)
- Raw Space-Track CDM fields viewer
- Live conjunction summary table with risk classification
- Per-CDM tabs: Risk Prediction | 3D Orbit Simulation | B-Plane View | Orbital Data
- Animated 3D conjunction approach visualization
- GP orbital enrichment for full simulation data

#### 3. Risk Assessment
Full conjunction analysis interface:
- Model selection and event selection controls
- Prediction results (risk value, category, confidence, 90% interval)
- Event details (event ID, time-to-TCA, miss distance, true risk)
- Physics verification (analytic Pc, covariance validation, short encounter check)
- Fused risk assessment (combined ML + physics result)
- AI explanation (natural language summary)
- Top risk drivers (SHAP waterfall chart)

#### 4. Event Timeline
Risk evolution tracking:
- Multi-CDM event selection
- Risk timeline chart (log10(Pc) + miss distance vs time-to-TCA)
- Trend analysis (increasing/decreasing/stable, convergence detection)
- CDM sequence table
- Escalating events detection (risk increasing above threshold)
- Risk matrix scatter plot (miss distance vs risk)

#### 5. Orbit Simulation
3D orbital visualization:
- Orbital environment view (multiple events around Earth globe)
- Conjunction detail view (single event with orbital elements)
- Interactive Plotly 3D globe with orbit traces
- Target and chaser orbit rendering with conjunction markers
- Covariance ellipsoid visualization
- RTN frame relative position breakdown

#### 6. Orbital Tracker
Animated satellite conjunction visualization:
- Catalog of representative satellites (ISS, Sentinel-2A, CryoSat-2, Starlink, COSMOS debris, Envisat)
- Kepler equation solver (Newton-Raphson) for orbital propagation
- Animated approach sequence with timeline slider
- Ground track projection
- Real-time miss distance and closing velocity displays

#### 7. Data Explorer
ESA Kelvins dataset browser:
- Dataset overview metrics (rows, columns, events, memory, null percentage)
- Sample rows (first 50, random 50, high-risk only)
- Column statistics (mean, std, percentiles, null counts)
- Feature distribution histograms and box plots
- Data quality assessment (outliers, negative values, event structure)
- Feature correlation heatmap (Pearson r)
- CDMs-per-event distribution analysis

#### 8. Model Performance
Training results and model comparison:
- Model comparison cards (RMSE, R², MAE per model, best model highlighted)
- Metric comparison bar charts (tabbed: RMSE, MAE, R², Correlation)
- Train vs validation performance table (overfitting gap analysis)
- Per-risk-band evaluation (separate metrics for HIGH/MEDIUM/LOW/NEGLIGIBLE)
- Binary classification metrics at −5.0 threshold (precision, recall, F1, AUC)

#### 9. Feature Importance
SHAP-based explainability:
- Global feature importance (top 10 from pipeline results)
- Interactive SHAP computation (configurable model and sample size)
- Mean |SHAP| bar chart for global ranking
- Individual prediction explanation (per-sample SHAP waterfall)
- Color-coded risk drivers (red = increases risk, blue = decreases risk)

#### 10. Ensemble Comparison
Stacking ensemble analytics:
- Individual model vs ensemble performance comparison
- Contribution weight analysis
- Ensemble advantage metrics
- Residual distribution comparison
- Per-band ensemble improvement analysis

#### 11. Physics Lab
First-principles collision probability analysis:
- Conjunction geometry (miss distance, relative speed, approach angle, encounter type)
- RTN decomposition (R, T, N components)
- Analytic collision probability with log10(Pc) result
- Covariance validation status and warnings
- Physical consistency checks (pass/fail)
- Monte Carlo verification comparison

#### 12. Alert Dashboard
Alert monitoring interface:
- Simulated real-time alert feed for representative satellites
- Severity breakdown (CRITICAL, WARNING, INFO)
- Timeline chart of alert activity
- Alert severity distribution analytics
- Top alerting objects ranking

#### 13. Report Generator
HTML report generation:
- Conjunction report (per-event analysis with physics and recommendations)
- Model evaluation report (training results summary)
- Event timeline report (risk evolution visualization)
- Downloadable HTML reports with styled output

#### 14. System Status
Infrastructure health dashboard:
- Platform information (OS, Python version, CPU, memory)
- Dependency status (all required packages with version)
- Module inventory (all orbital_sentinel subpackages)
- Model artifacts status (trained models, scaler, feature names)
- Dataset availability
- Test suite discovery

---

## 8. Database Layer

### Technology

**SQLAlchemy 2.0** ORM with support for SQLite (default, zero-config) and PostgreSQL (production).

### Schema (6 Tables)

#### CDMRecord
Stores ingested Conjunction Data Messages:
- `id`, `cdm_id` (unique), `event_id` (indexed)
- `source` (ESA/SpaceTrack), `created_at`, `tca`
- `miss_distance`, `relative_speed`, `risk`
- `sat1_name`, `sat2_name`
- `raw_data` (JSON), `mapped_data` (JSON)
- One-to-many relationship with PredictionRecord

#### PredictionRecord
Stores ML prediction results:
- `cdm_record_id` (FK), `model_name`
- `predicted_risk`, `risk_category`, `confidence`
- `interval_lower`, `interval_upper`
- `physics_log10_pc`, `fused_risk`
- `explanation` (text), `top_features` (JSON)
- `inference_time_ms`

#### EventRecord
Tracks conjunction event lifecycle:
- `event_id` (unique, indexed)
- `first_seen`, `last_updated`, `cdm_count`
- `latest_risk`, `peak_risk`, `risk_trend`
- `status` (ACTIVE/RESOLVED), `notes`

#### ModelRecord
Tracks trained model metadata:
- `model_name`, `model_type`, `trained_at`
- `train_rmse`, `val_rmse`, `train_r2`, `val_r2`
- `feature_count`, `artifact_path`
- `is_active` (current production model)
- `metadata_json` (arbitrary metadata)

#### AlertRecord
Stores alert history:
- `event_id` (indexed), `alert_type`, `severity`
- `title`, `detail`, `created_at`
- `acknowledged`, `acknowledged_at`

#### SchemaVersion
Tracks database migrations:
- `version`, `applied_at`, `description`

---

## 9. Monitoring and Alerts

### 9.1 Data Drift Detection

The monitoring module detects distribution shifts between training and new data using:

**Kolmogorov-Smirnov Test**:
- Computes KS statistic for each feature
- Drift detected when p-value < 0.05 (configurable)
- Reports per-feature drift status and severity

**Population Stability Index (PSI)**:
- Compares prediction distributions
- PSI > 0.2 indicates significant drift
- Combined with KS test for robustness

**Drift Severity Levels**:
- `none`: No features drifted
- `low`: < 20% of features drifted
- `medium`: 20–50% of features drifted
- `high`: > 50% of features drifted OR prediction distribution shifted

### 9.2 Alert Rules

Six predefined alert rules evaluate every prediction:

| Rule                   | Severity   | Trigger Condition                           |
|------------------------|------------|---------------------------------------------|
| High Risk              | CRITICAL   | Predicted risk > −5.0                       |
| Medium Risk            | WARNING    | Predicted risk between −7.0 and −5.0        |
| Low Confidence         | WARNING    | Prediction confidence < 0.3                 |
| Physics Disagreement   | WARNING    | ML and physics differ by > 3 orders of magnitude |
| Covariance Invalid     | WARNING    | Covariance validation failed                |
| Wide Interval          | INFO       | Prediction interval width > 5.0             |

### 9.3 Notification Channels

- **Log-based**: Structured logging of all alerts
- **Webhook**: HTTP POST to configurable endpoints (Slack, Teams, custom)
- **Database**: All alerts persisted in AlertRecord table

---

## 10. Testing Strategy

### Test Suite

16 test modules covering all major system components:

| Test Module           | Covers                                    |
|----------------------|-------------------------------------------|
| `test_models.py`     | Model training, prediction, serialization  |
| `test_physics.py`    | Collision probability, covariance, geometry|
| `test_features.py`   | Feature engineering correctness            |
| `test_preprocessing.py` | Cleaning, normalization, splitting      |
| `test_evaluation.py` | Metrics computation, calibration           |
| `test_fusion.py`     | Risk fusion, confidence, disagreement      |
| `test_uncertainty.py`| Prediction intervals, calibration          |
| `test_explainability.py` | SHAP analysis, explanations            |
| `test_validation.py` | Schema validation, quality checks          |
| `test_ingestion.py`  | Dataset loading, CDM mapping               |
| `test_api.py`        | FastAPI endpoint testing                   |
| `test_database.py`   | ORM models, repository operations          |
| `test_alerts.py`     | Alert rules, manager, notifier             |
| `test_reporting.py`  | HTML report generation                     |
| `test_cdm_api.py`    | Space-Track API client                     |
| `test_ml_advanced.py`| Ensemble, tuning, advanced ML              |

### CI Pipeline

GitHub Actions runs on every push and PR to `main`:

1. **Lint** (`ruff check src/`): Code style and error detection
2. **Test** (`pytest tests/ -v`): Full test suite with coverage
3. **Type Check** (`mypy`): Static type analysis (continue-on-error)
4. **Docker Build** (main branch only): Build and smoke test container

---

## 11. Deployment

### Docker Compose

Three-service architecture:

```yaml
services:
  api:          # FastAPI (port 8000) — REST API with health checks
  dashboard:    # Streamlit (port 8501) — Interactive dashboard
  pipeline:     # Training pipeline (on-demand, training profile)
```

Shared volumes for data, models, configs, and proofs directories. The API service includes a health check (`/health` endpoint) with the dashboard depending on API health.

### Quick Start

```bash
# Development
pip install -e ".[dev]"
python scripts/run_pipeline.py       # Train models
python -m orbital_sentinel.api       # Start API (port 8000)
streamlit run dashboard/app.py       # Start dashboard (port 8501)

# Docker
docker-compose up -d                 # API + Dashboard
docker-compose --profile training up # Include training pipeline
```

### Configuration

`configs/config.yaml` centralizes all settings:
- Data paths and file names
- Target column and floor value
- Feature column classifications (ID, target, leakage, categorical)
- Split parameters (test size, strategy, random state)
- Model save directory
- Physics parameters (combined radius, Monte Carlo samples)
- Dashboard host and port

---

## 12. Complete Feature List

### Data Ingestion
- ESA Kelvins dataset loader (162K CDMs, 102 features)
- Space-Track.org CDM Public API client with authentication
- General Perturbation (GP) orbital element enrichment
- CDM-to-ESA format mapper for standardized processing

### Data Processing
- Schema validation with type and range checking
- Data quality assessment (outliers, missing values, negative values)
- Automated data cleaning pipeline
- Feature scaling and normalization
- Event-based data splitting (prevents data leakage)
- Data leakage detection module

### Machine Learning
- XGBoost gradient boosted trees
- LightGBM gradient boosted trees
- Random Forest ensemble
- Ridge regression (L2 regularized)
- LSTM recurrent neural network (PyTorch)
- GRU recurrent neural network (PyTorch)
- Transformer encoder network (PyTorch)
- Stacking ensemble with RidgeCV meta-learner
- Optuna hyperparameter optimization (TPE sampler)
- Cross-validated model selection

### Feature Engineering
- 144 derived features from 102 raw columns
- Static CDM features (miss distance, speed, classification)
- Orbital mechanics features (Keplerian elements, energy, angular momentum)
- Covariance-derived features (eigenvalues, condition number, uncertainty volume)
- Temporal features (time-to-TCA, update frequency, risk change rate)

### Physics Engine
- 2D short-encounter collision probability (Foster/Akella-Alfriend)
- Foster analytic approximation (cross-check)
- Covariance matrix validation (symmetry, PSD, condition number)
- Covariance projection to encounter plane
- Monte Carlo collision probability verification (10,000 samples)
- Conjunction geometry analysis (RTN frame)
- Encounter type classification (head-on, crossing, co-orbital)
- Short-encounter assumption validation

### Risk Fusion
- Adaptive ML + physics weighted fusion
- Dynamic physics weight based on disagreement severity
- Covariance quality-aware trust adjustment
- Confidence scoring (interval width, agreement, covariance quality)
- ML-physics disagreement analysis with severity levels
- Prediction interval expansion when physics unavailable

### Explainability
- SHAP value computation for any model
- Global feature importance ranking
- Per-prediction feature contribution analysis
- Natural language explanation generation
- Top risk driver identification

### Uncertainty Quantification
- Prediction interval estimation
- Interval calibration
- Confidence scoring with multi-factor breakdown

### Evaluation
- Regression metrics (RMSE, MAE, R², correlation)
- Binary classification at configurable threshold
- Per-risk-band evaluation
- Calibration analysis
- K-fold cross-validation
- Train-validation gap (overfitting detection)
- Threshold optimization

### REST API
- Full prediction with physics and SHAP (POST /api/v1/predict)
- Quick prediction (ML only, POST /api/v1/predict/quick)
- Batch prediction (POST /api/v1/predict/batch)
- Live CDM fetch with optional predictions (POST /api/v1/cdm/fetch)
- Full CDM analysis (POST /api/v1/cdm/analyze)
- Dataset sampling (GET /api/v1/cdm/sample)
- Model management (list, info, features, details)
- Health check with uptime tracking
- Auto-generated Swagger/ReDoc documentation

### CLI
- Model training command
- Prediction command
- API server launcher
- Data management
- System status

### Dashboard (14 Pages)
- Mission Control (operational overview)
- Live CDM Feed (Space-Track real-time data)
- Risk Assessment (ML + physics analysis)
- Event Timeline (risk evolution tracking)
- Orbit Simulation (3D globe with orbits)
- Orbital Tracker (animated conjunction approach)
- Data Explorer (dataset browser and statistics)
- Model Performance (training results comparison)
- Feature Importance (SHAP explainability)
- Ensemble Comparison (stacking ensemble analytics)
- Physics Lab (first-principles Pc analysis)
- Alert Dashboard (alert monitoring)
- Report Generator (HTML reports)
- System Status (infrastructure health)

### Database
- SQLAlchemy 2.0 ORM (SQLite/PostgreSQL)
- 6 tables (CDMRecord, PredictionRecord, EventRecord, ModelRecord, AlertRecord, SchemaVersion)
- Schema versioning and migration support
- Repository pattern data access

### Monitoring
- Feature drift detection (KS test)
- Prediction drift detection (KS + PSI)
- Multi-level severity classification
- Data quality monitoring

### Alerting
- 6 configurable alert rules
- CRITICAL/WARNING/INFO severity levels
- Webhook notifications (Slack, Teams)
- Log-based alerting
- Alert persistence and acknowledgment

### Visualization
- Interactive 3D orbit rendering (Plotly)
- Earth globe with orbit traces
- Conjunction geometry visualization
- Covariance ellipsoid rendering
- B-plane close approach view
- Animated conjunction approach sequences
- Dark glassmorphism theme with consistent design system

### Reporting
- Conjunction analysis HTML reports
- Model evaluation reports
- Event timeline reports
- Downloadable styled HTML output

### DevOps
- Docker multi-service deployment (API + Dashboard + Pipeline)
- GitHub Actions CI/CD (lint, test, type-check, Docker build)
- YAML-based configuration management
- Reproducible training with fixed random seeds

---

*Document generated for Orbital Sentinel v2.0 — ML-Powered Satellite Conjunction Risk Assessment System*
