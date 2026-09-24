# ORBITAL SENTINEL

## Automated Machine Learning Triage System for Space Debris Conjunction Analysis

**Project Type:** AI/ML + Physics-Based Space Safety Analysis  
**Primary Dataset:** ESA Kelvins Collision Avoidance Challenge Dataset  
**Primary Model:** XGBoost Regression  
**Interface:** Streamlit + Plotly  
**Explainability:** SHAP  
**Validation:** Event-Based Validation + Physics Verification + Automated Testing

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. What Is a Conjunction?](#2-what-is-a-conjunction)
- [3. Dataset](#3-dataset)
- [4. Raw Feature Categories](#4-raw-feature-categories)
- [5. Feature Engineering](#5-feature-engineering)
- [6. Data Preprocessing](#6-data-preprocessing)
- [7. Models and Results](#7-models-and-results)
- [8. Physics Engine](#8-physics-engine)
- [9. Explainability (SHAP)](#9-explainability-shap)
- [10. System Architecture](#10-system-architecture)
- [11. Dashboard](#11-dashboard)
- [12. REST API and CLI](#12-rest-api-and-cli)
- [13. Technology Stack](#13-technology-stack)
- [14. Project Structure](#14-project-structure)
- [15. Quick Start](#15-quick-start)
- [16. Testing and Quality](#16-testing-and-quality)
- [17. Future Development](#17-future-development)
- [18. License](#18-license)

---

## 1. Project Overview

Space debris poses a growing threat to operational satellites and crewed spacecraft. Over 34,000 objects larger than 10 cm are tracked in low-Earth orbit, generating hundreds of conjunction alerts daily. Satellite operators must rapidly assess collision risk from Conjunction Data Messages (CDMs) to decide whether evasive maneuvers are warranted.

**Orbital Sentinel** automates this triage process. The system combines:

1. Historical conjunction data (ESA Kelvins dataset)
2. Data preprocessing and cleaning
3. Domain-specific feature engineering
4. Machine learning (XGBoost, LightGBM, Random Forest, Ridge, LSTM, GRU, Transformer)
5. Physics-based verification (analytic Pc, Monte Carlo simulation)
6. Uncertainty estimation (prediction intervals, calibration)
7. Explainable AI (SHAP feature attribution)
8. Interactive visualization (14-page Streamlit dashboard)

The primary machine-learning task is **regression**. The model predicts:

```
log10(Pc)
```

where `Pc` is the collision probability and `log10(Pc)` is its base-10 logarithm. A less-negative value represents a higher collision probability.

### Risk Classification

| Category    | log10(Pc) Range | Interpretation                       |
|-------------|----------------|--------------------------------------|
| HIGH        | > -5           | Potential maneuver candidate         |
| MEDIUM      | -7 to -5       | Elevated risk, increased monitoring  |
| LOW         | -15 to -7      | Standard tracking, no action needed  |
| NEGLIGIBLE  | < -15          | Effectively zero risk                |

### Global Space Debris Population

| Type         | Occupancy | Count  |
|--------------|-----------|--------|
| Payloads     | 59%       | 18,697 |
| Debris       | 31%       | 9,907  |
| Rocket Bodies| 6%        | 2,108  |
| Unknown      | 2%        | 661    |

---

## 2. What Is a Conjunction?

A conjunction is a predicted close approach between two space objects. A conjunction does **not** automatically mean a collision. The system evaluates each encounter using:

- Miss distance (predicted separation at closest approach)
- Relative velocity (closing speed)
- Time to TCA (Time of Closest Approach)
- Relative position and velocity in RTN (Radial, Transverse, Normal) coordinates
- Covariance and uncertainty estimates
- Orbital parameters (semi-major axis, eccentricity, inclination)
- Observation quality (OD span, residuals, observation count)
- Object characteristics (type, radar cross-section, area-to-mass ratio)

```
Object 1
     \
      \
       \       Closest Approach
        \          *
         \        /
          \      /
           \    /
            \  /
             \/
             /\
            /  \
           /    \
Object 2
```

**TCA (Time of Closest Approach)** is the predicted time when the two objects reach their minimum separation. All conjunction quantities are evaluated around TCA.

---

## 3. Dataset

Orbital Sentinel uses the **ESA Kelvins Collision Avoidance Challenge** dataset.

### Dataset Statistics

| Property              | Value          |
|-----------------------|----------------|
| Training CDMs         | 162,634        |
| Testing CDMs          | 24,484         |
| Unique Events         | 13,154         |
| Raw Columns           | 103            |
| Retained Raw Features | 98             |
| Engineered Features   | 46             |
| Total Model Features  | 144            |
| Target Variable       | `risk` = log10(Pc) |
| Floor Value           | -30.0          |

Each row corresponds to a single CDM. Multiple CDMs can belong to the same conjunction event, forming a time series:

```
Event
 |-- CDM 1   (earliest observation)
 |-- CDM 2   (updated observation)
 |-- CDM 3   (updated observation)
 |-- CDM 4   (closest to TCA)
 +-- ...
```

### Risk Distribution

| Risk Level  | Threshold        | Count  | Percentage |
|-------------|------------------|--------|------------|
| HIGH        | log10(Pc) > -5   | 3,381  | 2.1%       |
| MEDIUM      | -7 to -5         | 17,782 | 10.9%      |
| LOW         | -15 to -7        | 53,575 | 32.9%      |
| NEGLIGIBLE  | < -15            | 87,896 | 54.1%      |

The dataset is heavily imbalanced: only 2.1% of CDMs are high-risk, reflecting real-world conditions where most conjunctions are safe.

### Why 98 Raw Model Features?

Five columns are excluded from the model input:

```
103 raw columns
 - 2 identifiers (event_id, mission_id)
 - 1 target (risk)
 - 2 leakage-related fields (max_risk_estimate, max_risk_scaling)
 = 98 retained raw features
```

---

## 4. Raw Feature Categories

The 103 columns in the ESA dataset are grouped into the following categories:

### 4.1 Event and Target Information

`event_id`, `mission_id`, `risk`, `time_to_tca`

### 4.2 Relative Geometry

| Feature                | Description                                |
|------------------------|--------------------------------------------|
| `miss_distance`        | Predicted separation at TCA (km)           |
| `relative_speed`       | Relative velocity magnitude at TCA (km/s)  |
| `relative_position_r`  | Radial separation component                |
| `relative_position_t`  | Along-track separation component           |
| `relative_position_n`  | Cross-track separation component           |
| `relative_velocity_r`  | Radial velocity component                  |
| `relative_velocity_t`  | Along-track velocity component             |
| `relative_velocity_n`  | Cross-track velocity component             |
| `azimuth`, `elevation` | Encounter geometry angles                  |
| `geocentric_latitude`  | Latitude of conjunction point              |

### 4.3 Orbital State (for both target `t` and chaser `c`)

`x_j2k_sma` (semi-major axis), `x_j2k_ecc` (eccentricity), `x_j2k_inc` (inclination), `x_h_apo` (apogee height), `x_h_per` (perigee height)

### 4.4 Covariance and Uncertainty

Position uncertainties (`x_sigma_r`, `x_sigma_t`, `x_sigma_n`), velocity uncertainties (`x_sigma_rdot`, `x_sigma_tdot`, `x_sigma_ndot`), and cross-correlation terms (`x_ct_r`, `x_cn_r`, `x_crdot_r`, etc.) for both target and chaser objects. `x_position_covariance_det` provides the determinant of the position covariance matrix.

### 4.5 Observation / Orbit Determination

`x_actual_od_span`, `x_recommended_od_span`, `x_obs_available`, `x_obs_used`, `x_residuals_accepted`, `x_weighted_rms`, `x_time_lastob_start`, `x_time_lastob_end`

### 4.6 Object Information

`c_object_type`, `x_rcs_estimate`, `x_cd_area_over_mass`, `x_cr_area_over_mass`, `x_sedr`

### 4.7 Space Weather

`F10` (solar flux), `F3M` (3-month average solar flux), `SSN` (sunspot number), `AP` (geomagnetic index)

---

## 5. Feature Engineering

The feature engineering pipeline transforms 98 raw features into a 144-dimensional input matrix:

```
98 raw features + 46 engineered features = 144 model features
```

### 5.1 Static / Geometry-Derived Features

| Feature                        | Formula / Description                          |
|--------------------------------|------------------------------------------------|
| `relative_position_magnitude`  | sqrt(R^2 + T^2 + N^2)                         |
| `relative_velocity_magnitude`  | Magnitude of velocity vector                   |
| `encounter_duration_proxy`     | miss_distance / relative_speed                 |
| `t_obs_utilization`            | obs_used / obs_available (target)              |
| `c_obs_utilization`            | obs_used / obs_available (chaser)              |
| `t_od_span_ratio`              | actual_od_span / recommended_od_span (target)  |
| `c_od_span_ratio`              | actual_od_span / recommended_od_span (chaser)  |
| `miss_to_mahalanobis_ratio`    | miss_distance / mahalanobis_distance           |

### 5.2 Orbital-Derived Features

For both target and chaser objects:

| Feature                   | Description                                      |
|---------------------------|--------------------------------------------------|
| `x_orbital_period`        | Derived from semi-major axis (Kepler's 3rd law)  |
| `x_mean_motion`           | Orbital angular rate                             |
| `x_perigee_alt`           | Perigee altitude above Earth surface             |
| `x_apogee_alt`            | Apogee altitude above Earth surface              |
| `x_orbit_energy`          | Specific orbital energy                          |
| `sma_difference`          | |target_sma - chaser_sma|                        |
| `inclination_difference`  | |target_inc - chaser_inc|                        |
| `x_orbit_eccentricity_proxy` | Derived eccentricity indicator                |

### 5.3 Covariance-Derived Features

| Feature                        | Description                                    |
|--------------------------------|------------------------------------------------|
| `x_position_uncertainty`       | Combined position sigma                        |
| `x_sigma_max`, `x_sigma_min`  | Max/min of position sigmas                     |
| `x_sigma_ratio`                | sigma_max / sigma_min (uncertainty anisotropy)  |
| `combined_position_uncertainty`| RSS of target and chaser position uncertainties|
| `combined_sigma_r/t/n`         | Combined RTN uncertainties                     |
| `miss_distance_sigma_ratio`    | **Key feature**: miss_distance / combined_uncertainty |
| `x_log_cov_det`               | Log of covariance determinant                  |

The `miss_distance_sigma_ratio` was the top SHAP feature. It tells the model how large the predicted separation is relative to the uncertainty scale, which is more informative than distance alone.

### 5.4 Temporal Features

| Feature                 | Description                                    |
|-------------------------|------------------------------------------------|
| `time_to_tca_hours`     | Time to TCA in hours                           |
| `time_to_tca_squared`   | Quadratic time term                            |
| `log_time_to_tca`       | Logarithmic time scale                         |
| `is_close_approach`     | Binary: near TCA indicator                     |
| `cdm_sequence_number`   | CDM position within event                      |
| `cdm_total_in_event`    | Total CDMs in event                            |
| `cdm_sequence_fraction` | Relative position in event timeline            |
| `x_observation_span`    | Duration of observation data                   |

### 5.5 Mahalanobis Distance

Mahalanobis distance measures separation relative to the uncertainty distribution:

```
D = sqrt((x - mu)^T * Sigma^-1 * (x - mu))
```

This is more informative than Euclidean distance when covariance structures are anisotropic.

---

## 6. Data Preprocessing

### 6.1 Cleaning Pipeline

1. Replace infinite values with NaN
2. Remove all-NaN columns
3. Encode categorical fields (`c_object_type`) numerically
4. Remove duplicate rows
5. Impute remaining missing values with column medians

### 6.2 Normalization

StandardScaler is applied to all features:

```
z = (x - mean) / standard_deviation
```

The scaler is fitted on training data only and applied to validation data. The trained scaler is saved as `models_saved/scaler.joblib`.

### 6.3 Data Leakage Prevention

The following columns are excluded from model input:

- `event_id`, `mission_id` (identifiers)
- `risk` (target variable)
- `max_risk_estimate`, `max_risk_scaling` (target-derived information)

An explicit leakage check validates that no target-correlated features enter the feature matrix.

### 6.4 Event-Based Train/Validation Split

A random row split would allow CDMs from the same event to appear in both sets. Instead, the project splits by **event ID**:

```
Training events:    10,524
Validation events:   2,630
Event overlap:           0
```

This ensures the model is evaluated on entirely unseen conjunction events.

---

## 7. Models and Results

### 7.1 Models Used

| Model             | Type                    | Configuration                                              |
|-------------------|-------------------------|------------------------------------------------------------|
| Ridge Regression  | Linear baseline         | L2 regularization                                          |
| Random Forest     | Ensemble (bagging)      | 200 trees, max_depth=20, n_jobs=-1                         |
| LightGBM          | Gradient boosting       | 500 trees, depth=8, lr=0.05, num_leaves=63, subsample=0.8  |
| **XGBoost**       | **Gradient boosting**   | **500 trees, depth=8, lr=0.05, subsample=0.8, colsample=0.8** |
| LSTM              | Temporal deep learning  | Sequence model for CDM time series                         |
| GRU               | Temporal deep learning  | Lighter recurrent alternative to LSTM                      |
| Transformer       | Temporal deep learning  | Self-attention over CDM sequences                          |

The primary completed model is **XGBoost**, which reached `best_iteration = 499` with early stopping configured at 50 rounds.

### 7.2 Validation Results

| Model         | R^2    | RMSE  | MAE   | Accuracy | F1    | Precision | Recall | AUC   | Train Time |
|---------------|--------|-------|-------|----------|-------|-----------|--------|-------|------------|
| **XGBoost**   | **0.921** | **2.82** | **1.55** | **86.9%** | **86.3%** | **86.3%** | **86.9%** | **0.967** | 27 s |
| LightGBM      | 0.920  | 2.84  | 1.63  | 86.2%    | 85.5% | 85.4%     | 86.2%  | 0.962 | 12 s       |
| Random Forest | 0.892  | 3.30  | 1.79  | 83.9%    | 82.6% | 83.6%     | 83.9%  | 0.959 | 529 s      |
| Ridge         | 0.521  | 6.93  | 5.46  | 66.9%    | 62.7% | 62.5%     | 66.9%  | 0.654 | 1.3 s      |

**XGBoost explains 92.1% of variance in collision probability** (R^2 = 0.921) with an AUC of 0.967 for risk-tier classification.

### 7.3 Why XGBoost?

Collision-risk data contains nonlinear relationships where miss distance, relative velocity, uncertainty, and orbital geometry interact in complex ways. XGBoost's gradient-boosted decision trees can learn these nonlinear structures and feature interactions effectively on structured tabular data, outperforming both the linear baseline (R^2 = 0.521) and Random Forest (R^2 = 0.892).

### 7.4 High-Risk Binary Evaluation

Using `log10(Pc) >= -5.0` as the high-risk threshold:

```
Precision:  0.698
Recall:     0.423
F1:         0.527
ROC-AUC:    0.967
```

### 7.5 Additional Metrics

| Metric                          | Value  |
|---------------------------------|--------|
| Correlation (predicted vs actual)| 0.960 |
| Median Absolute Error           | 0.651  |
| Max Error                       | 21.82  |
| Expected Calibration Error (ECE)| 0.032  |
| Brier Score                     | 0.087  |

### 7.6 Hyperparameter Tuning

Optuna-based hyperparameter search is implemented in `src/orbital_sentinel/models/tuning.py` with search spaces:

- XGBoost: 200-1000 estimators, depth 4-12, lr 0.01-0.3
- LightGBM: 200-1000 estimators, depth 4-12, 20-127 leaves
- Random Forest: 100-600 estimators, depth 6-20

### 7.7 Stacking Ensemble

A stacking ensemble with RidgeCV meta-learner combines base model predictions for improved robustness.

---

## 8. Physics Engine

The ML model is complemented by an independent physics-based analysis layer.

### 8.1 Analytic Collision Probability

- **Foster / Akella-Alfriend method**: 2D short-encounter collision probability calculation
- Uses combined position covariance in the encounter plane
- Requires miss distance, combined covariance, and combined object hard-body radius

### 8.2 Monte Carlo Simulation

- 10,000 sample propagation through orbital uncertainty
- Provides empirical collision probability estimate
- Independent validation of analytic Pc results

### 8.3 Covariance Validation

- Positive-definite matrix checks
- Eigenvalue analysis
- Anisotropy detection

### 8.4 Risk Fusion

The system fuses ML predictions with physics results:

```
ML Prediction (log10 Pc)
         +
Physics Analytic Pc
         +
Confidence Scoring
         +
Disagreement Detection
         |
         v
Fused Risk Assessment
```

When ML and physics predictions disagree significantly, the system flags the case for operator review.

### 8.5 Physics Verification

```
Physics validation sample: 20/20 successful
```

All 20 sampled validation conjunctions produced valid analytic Pc outputs.

---

## 9. Explainability (SHAP)

SHAP (SHapley Additive exPlanations) provides feature-level attribution for each prediction:

```
144 features --> XGBoost --> prediction --> SHAP --> feature contributions
```

SHAP values were computed on 200 validation samples. Top features:

| Rank | Feature                 | SHAP Value | Physical Meaning                            |
|------|-------------------------|------------|---------------------------------------------|
| 1    | `miss_distance`         | 0.348      | Closest approach distance between objects    |
| 2    | `time_to_tca`           | 0.319      | Time remaining until closest approach        |
| 3    | `relative_speed`        | 0.275      | Closing velocity at TCA                      |
| 4    | `mahalanobis_distance`  | 0.232      | Statistical distance accounting for covariance|
| 5    | `t_j2k_sma`            | 0.183      | Target object semi-major axis (altitude)     |
| 6    | `t_j2k_ecc`            | 0.143      | Target orbit eccentricity                    |
| 7    | `c_j2k_sma`            | 0.101      | Chaser object semi-major axis                |

Miss distance being the top predictor aligns with orbital mechanics: closer passes directly increase collision probability.

The explainability module also generates natural-language explanations, translating SHAP values into operator-readable risk narratives.

---

## 10. System Architecture

### 10.1 Overall Technical Flow

```
                 ESA KELVINS DATASET
                         |
             +-----------+-----------+
             |                       |
        ESA KELVINS             SPACE-TRACK
        Historical               Live CDM API
             |                       |
             +-----------+-----------+
                         |
                  DATA INGESTION
                         |
               VALIDATION / CLEANING
                         |
              FEATURE ENGINEERING
                         |
                    144 FEATURES
                         |
               EVENT-BASED SPLIT
                         |
          +--------------+---------------+
          |              |               |
       XGBoost       Temporal         Physics
       LightGBM      Models           Engine
       RF / Ridge    (LSTM/GRU/       (Analytic Pc,
                      Transformer)     Monte Carlo)
          |              |               |
          +--------------+---------------+
                         |
                  STACKING ENSEMBLE
                         |
                    RISK FUSION
                         |
            UNCERTAINTY QUANTIFICATION
                         |
               SHAP EXPLAINABILITY
                         |
                  DECISION SUPPORT
                         |
          +----------+----------+----------+
          |          |          |          |
       REST API   CLI Tool   Dashboard   Alerts
       (FastAPI)  (Click)    (Streamlit) (Webhook)
```

### 10.2 ML + Physics Fusion Architecture

```
            CONJUNCTION DATA
                   |
            PREPROCESSING
                   |
            144 FEATURES
                   |
          +--------+--------+
          |                 |
       XGBoost        Physics Engine
          |                 |
    ML Risk Estimate   Analytic Pc
          |                 |
          +--------+--------+
                   |
          +--------+--------+
          |                 |
     Risk Fusion      Uncertainty
          |           Intervals
          +--------+--------+
                   |
            SHAP Analysis
                   |
         Streamlit Dashboard
```

---

## 11. Dashboard

The Streamlit dashboard provides an interactive 14-page interface with a dark glassmorphism theme and Plotly visualizations.

### Dashboard Pages

| Page                 | Description                                        |
|----------------------|----------------------------------------------------|
| Mission Control      | Overview with real-time risk metrics and status     |
| Data Explorer        | CDM dataset exploration with risk-tier visualization|
| Feature Analysis     | Feature importance, distributions, correlations     |
| Model Training       | Training pipeline status and configuration          |
| Model Comparison     | Side-by-side model performance metrics              |
| Risk Assessment      | Per-event collision probability prediction          |
| Orbit Simulation     | 3D orbital environment with conjunction geometry    |
| Physics Engine       | Physics-based collision probability analysis        |
| CDM Feed             | Real-time CDM data retrieval and analysis           |
| Ensemble Comparison  | Radar charts, model weightage, ensemble analytics   |
| Alert Dashboard      | Configurable alert rules and notification history   |
| Report Generator     | Downloadable reports (Markdown, HTML, JSON)         |
| System Status        | Model health, data quality, infrastructure metrics  |
| Orbital Tracker      | Animated conjunction visualization with B-plane view|

### Running the Dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
```

---

## 12. REST API and CLI

### 12.1 FastAPI REST API

12 endpoints on port 8000:

| Endpoint Group | Endpoints                                    | Description              |
|---------------|----------------------------------------------|--------------------------|
| Predictions   | `/predict`, `/predict/quick`, `/predict/batch`| Risk prediction          |
| CDM           | `/cdm/fetch`, `/cdm/analyze`, `/cdm/sample`  | CDM operations           |
| Health        | `/health`, `/status`                          | System health checks     |
| Models        | `/models`, `/models/{name}`, `/models/features`, `/models/metrics` | Model info |

```bash
python -m orbital_sentinel.api  # Start API server
```

### 12.2 Click CLI

| Command   | Description                            |
|-----------|----------------------------------------|
| `train`   | Run model training pipeline            |
| `predict` | Predict risk for a CDM or event        |
| `serve`   | Start the FastAPI server               |
| `data`    | Data management and ingestion          |
| `status`  | System status and diagnostics          |

```bash
orbital-sentinel --help
orbital-sentinel predict --event-id 1234
```

---

## 13. Technology Stack

| Technology    | Purpose                                |
|---------------|----------------------------------------|
| Python 3.11   | Core implementation                    |
| Pandas        | Data processing and manipulation       |
| NumPy         | Numerical computation                  |
| Scikit-learn  | Preprocessing, scaling, metrics        |
| XGBoost       | Primary gradient boosting model        |
| LightGBM      | Alternative gradient boosting model    |
| PyTorch       | Temporal models (LSTM, GRU, Transformer)|
| SHAP          | Explainable AI / feature attribution   |
| Plotly        | Interactive 2D/3D visualization        |
| Streamlit     | Dashboard application framework        |
| FastAPI       | REST API server                        |
| Click         | Command-line interface                 |
| SQLAlchemy    | Database ORM (SQLite/PostgreSQL)       |
| Optuna        | Hyperparameter optimization            |
| SciPy         | Statistical tests, physics calculations|
| Pytest        | Automated testing                      |
| Pylint        | Static code analysis                   |
| Docker        | Containerization                       |
| Space-Track   | Live CDM data source                   |

---

## 14. Project Structure

```
Orbital-Sentinel-PBL/
|
|-- data/
|   +-- raw/esa_kelvins/           # ESA Kelvins dataset (not in git)
|
|-- dashboard/
|   |-- app.py                     # Main dashboard (14 pages)
|   +-- _pages/
|       |-- orbital_tracker.py     # Animated conjunction visualization
|       |-- ensemble_comparison.py # Model comparison analytics
|       |-- alert_dashboard.py     # Alert monitoring
|       |-- report_generator.py    # Report generation with downloads
|       +-- system_status.py       # Infrastructure health
|
|-- docs/
|   |-- architecture.md            # Architecture documentation
|   |-- TECHNICAL_DOCUMENTATION.md # Full technical documentation
|   +-- PRESENTATION_CONTENT.md    # Presentation slide content
|
|-- models_saved/
|   |-- XGBoost_model.joblib       # Trained XGBoost model (6.6 MB)
|   |-- LightGBM_model.joblib      # Trained LightGBM model (2.8 MB)
|   |-- RandomForest_model.joblib  # Trained Random Forest (175 MB)
|   |-- Logistic_model.joblib      # Trained Ridge/Logistic (1.8 KB)
|   |-- scaler.joblib              # Fitted StandardScaler
|   |-- feature_names.json         # 144 feature names
|   +-- model_metrics.json         # Evaluation metrics for all models
|
|-- proofs/                        # Screenshots and progress logs
|
|-- scripts/
|   +-- run_pipeline.py            # 11-step training pipeline
|
|-- src/orbital_sentinel/
|   |-- api/                       # FastAPI REST API (12 endpoints)
|   |   |-- app.py
|   |   |-- schemas.py
|   |   +-- routes/ (predictions, cdm, health, models)
|   |-- cli/                       # Click CLI (5 commands)
|   |   +-- commands/ (train, predict, serve, data, status)
|   |-- config/                    # YAML-based settings
|   |-- database/                  # SQLAlchemy (6 tables)
|   |-- evaluation/                # Metrics, calibration, cross-validation
|   |-- events/                    # Event reconstruction, sequencing
|   |-- explainability/            # SHAP analysis, NL explanations
|   |-- features/                  # Static, orbital, covariance, temporal
|   |-- fusion/                    # ML + physics risk fusion
|   |-- inference/                 # Single event and batch prediction
|   |-- ingestion/                 # Dataset loader, Space-Track API
|   |-- models/
|   |   |-- baselines/             # XGBoost, LightGBM, RF, Logistic
|   |   |-- temporal/              # LSTM, GRU, Transformer
|   |   |-- ensemble.py            # Stacking ensemble
|   |   +-- tuning.py              # Optuna hyperparameter search
|   |-- monitoring/                # Data quality, drift detection
|   |-- physics/                   # Collision Pc, Monte Carlo, covariance
|   |-- preprocessing/             # Cleaning, normalization, splitting
|   |-- reporting/                 # HTML report generation
|   |-- alerts/                    # Alert rules, webhook notifications
|   |-- uncertainty/               # Prediction intervals, calibration
|   |-- validation/                # Schema, quality, leakage checks
|   +-- visualization/             # 3D Plotly orbit rendering
|
|-- tests/                         # 16 test modules
|
|-- configs/config.yaml            # Project configuration
|-- docker-compose.yml             # Multi-service deployment
|-- requirements.txt               # Python dependencies
|-- .pylintrc                      # Pylint configuration
+-- setup.py                       # Package setup
```

---

## 15. Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gokul2736/Orbital-Sentinel-PBL.git
cd Orbital-Sentinel-PBL

# Install with development dependencies
pip install -e ".[dev]"
```

### Training Pipeline

```bash
# Run the full 11-step pipeline
python scripts/run_pipeline.py
```

### Dashboard

```bash
# Start the Streamlit dashboard (port 8501)
streamlit run dashboard/app.py
```

### API Server

```bash
# Start the FastAPI server (port 8000)
python -m orbital_sentinel.api
```

### CLI

```bash
# View available commands
orbital-sentinel --help

# Predict risk for an event
orbital-sentinel predict --event-id 1234
```

### Docker

```bash
# Start the full stack
docker-compose up -d
```

### Tests

```bash
# Run the full test suite
pytest tests/ -v
```

---

## 16. Testing and Quality

### Test Results

```
Tests:    81 passed, 0 failed
Modules:  16 test files
Pylint:   9.85 / 10
```

### Test Coverage

| Module            | Tests Cover                          |
|-------------------|--------------------------------------|
| test_models.py    | Model training and prediction        |
| test_physics.py   | Collision probability calculations   |
| test_features.py  | Feature engineering pipeline         |
| test_preprocessing.py | Data cleaning and normalization  |
| test_evaluation.py| Metrics computation                  |
| test_fusion.py    | ML + physics risk fusion             |
| test_uncertainty.py | Prediction intervals               |
| test_explainability.py | SHAP analysis                  |
| test_validation.py| Schema and quality checks            |
| test_ingestion.py | Data loading                         |
| test_api.py       | REST API endpoints                   |
| test_database.py  | Database operations                  |
| test_alerts.py    | Alert rules and notifications        |
| test_reporting.py | Report generation                    |
| test_cdm_api.py   | Space-Track API client               |
| test_ml_advanced.py | Ensemble and advanced models       |

---

## 17. Future Development

- Continuous live Space-Track monitoring and automated alerting
- Full historical CDM reconstruction for post-event analysis
- High-fidelity orbital propagation (SGP4/SDP4)
- 3D trajectory replay with temporal scrubbing
- Advanced temporal modelling with trained LSTM/GRU/Transformer
- Physics-informed machine learning (hybrid loss functions)
- Multi-source surveillance-data fusion
- Multi-satellite constellation analysis
- Automated sensitivity / what-if analysis
- Probabilistic scenario simulation
- Dynamic uncertainty updates as new CDMs arrive
- Human-in-the-loop operator feedback integration
- Large-scale catalogue monitoring (>100K objects)
- More extensive ablation experiments across feature groups
- Real-time deployment with sub-second inference

---

## 18. License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Verified Results Summary

```
Training rows:              162,634
Testing rows:                24,484

Raw columns:                    103
Retained raw features:           98
Engineered features:             46
Final model features:           144

Training events:             10,524
Validation events:            2,630
Event overlap:                    0

XGBoost validation R^2:       0.921
LightGBM validation R^2:      0.920
Random Forest validation R^2:  0.892
Ridge validation R^2:          0.521

XGBoost RMSE:                  2.82
XGBoost AUC:                  0.967
XGBoost best iteration:         499

High-risk Precision:          0.698
High-risk Recall:             0.423
High-risk F1:                 0.527

Physics verification:        20/20 successful
SHAP sample size:               200
Automated tests:         81 passed
Pylint score:             9.85 / 10
```
