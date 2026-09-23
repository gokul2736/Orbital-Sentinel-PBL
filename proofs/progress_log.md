# Orbital Sentinel — Progress Log

## Phase 1: Data Pipeline
- Implemented ESA Kelvins dataset loader with train/test split support
- Built schema validation and data quality assessment modules
- Created preprocessing pipeline: cleaning, normalization, event-based splitting
- Engineered features across 4 categories: static, orbital, covariance, temporal
- Added leakage detection to prevent target leakage in feature set
- Result: 162,634 CDMs processed with 102 features

## Phase 2: ML Models
- Trained 4 baseline models: XGBoost, LightGBM, Random Forest, Logistic Regression
- XGBoost achieved best validation RMSE of 2.8175
- Built evaluation module with per-risk-band metrics and binary classification at -5.0 threshold
- Added prediction uncertainty estimation with 90% confidence intervals
- Implemented model save/load with joblib serialization

## Phase 3: Physics Engine
- Built analytic collision probability calculator (Alfriend method)
- Implemented covariance matrix validation (positive definiteness, symmetry)
- Created conjunction geometry analysis (RTN frame, approach angle, encounter duration)
- Added Monte Carlo verification for cross-checking analytic Pc
- Built physical consistency checks (Keplerian bounds, energy conservation)

## Phase 4: Risk Fusion & Explainability
- Developed ML + physics risk fusion with weighted combination
- SHAP-based feature importance (global and per-prediction)
- Natural language explanation generator for conjunction assessments
- Decision summary with recommended actions based on risk level
- Confidence scoring that accounts for model agreement and data quality

## Phase 5: Dashboard
- Built 9-page Streamlit dashboard with dark glassmorphism theme
- Pages: Mission Control, Risk Assessment, Event Timeline, Orbit Simulation, Live CDM Feed, Data Explorer, Model Performance, Feature Importance, Physics Lab
- 3D orbit visualization with Plotly (Earth globe, conjunction geometry, B-plane)
- Real-time CDM fetch from Space-Track.org with live predictions
- Interactive SHAP analysis and model comparison

## Phase 6: API & Infrastructure
- FastAPI REST API with prediction, CDM ingestion, model management endpoints
- SQLAlchemy database layer (CDM, prediction, event, alert, model tables)
- Click CLI tool with train/predict/serve/data/status commands
- HTML report generator for conjunction assessments and model performance
- Alert system with configurable rules and Slack webhook notifications
- Docker multi-stage build with docker-compose stack
- GitHub Actions CI/CD pipeline
- pyproject.toml packaging with console script entry point

## Phase 7: Advanced ML & Extended Dashboard
- Stacking ensemble model combining XGBoost, LightGBM, RF, Ridge via RidgeCV meta-learner
- Event-based K-fold cross-validation to prevent data leakage
- Optuna hyperparameter tuning for XGBoost, LightGBM, Random Forest
- SHAP-based and variance/correlation feature selection
- Binary threshold optimization (F1, F2, precision, recall targets)
- Multi-threshold evaluation at operational decision points
- Dashboard expanded from 9 to 13 pages:
  - Ensemble Comparison: model radar chart, per-model scatter/residual tabs, architecture diagram, weight analysis
  - Alert Dashboard: active alert panel with severity, timeline scatter, donut/stacked breakdown, rule config
  - Report Generator: conjunction/model/timeline reports with HTML preview and export
  - System Status: health monitoring, dependency check, module inventory, performance gauges

## Test Coverage
- 148 tests across 16 test modules
- Coverage: ingestion, preprocessing, features, models, evaluation, physics, fusion, uncertainty, explainability, API, database, reporting, alerts, ensemble, cross-validation, threshold optimization
