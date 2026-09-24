# Orbital Sentinel — Final Project Presentation Content

> Slide-by-slide guide for the final project review presentation.
> Each slide includes **Title**, **Key Points** (for the slide itself), **Speaker Notes** (what to say aloud), and **Visual Suggestion** (what to display).

---

## Slide 1: Title Slide

**Title:** Orbital Sentinel: ML-Powered Satellite Conjunction Risk Assessment

**Key Points:**
- Automated Machine Learning Triage System for Space Debris Conjunction Analysis
- [Your Name / Team Name]
- [Course Name / Institution]
- [Date]

**Speaker Notes:**
Good [morning/afternoon]. I'm presenting Orbital Sentinel — an end-to-end machine learning system that predicts satellite collision risk from real conjunction data messages. The system fuses ML predictions with physics-based analysis and delivers results through an interactive dashboard designed for mission operators. It processes real ESA conjunction data and produces explainable, uncertainty-aware risk assessments.

**Visual Suggestion:** Dark-themed title card with a stylized orbital trajectory graphic. Satellite silhouettes with crossing orbits in the background.

---

## Slide 2: The Space Debris Crisis

**Title:** Why This Matters — The Kessler Syndrome Threat

**Key Points:**
- 34,000+ tracked objects in Low Earth Orbit (LEO)
- Payloads: 59% (18,697) | Debris: 31% (9,907) | Rocket Bodies: 6% (2,108) | Unknown: 2% (661)
- Kessler Syndrome: cascading collisions generating exponentially more debris
- Current manual conjunction screening is slow, subjective, and doesn't scale
- Critical infrastructure (GPS, communications, weather) depends on safe orbits

**Speaker Notes:**
Space is getting crowded. There are over 34,000 tracked objects in orbit, and roughly a third of those are debris. The Kessler Syndrome describes a tipping point where collisions produce debris that causes more collisions — a cascade that could make certain orbits unusable. Right now, conjunction screening is largely manual, requiring analysts to review thousands of close-approach warnings daily. That process doesn't scale. Orbital Sentinel was built to automate and enhance that triage process using machine learning.

**Visual Suggestion:** Pie chart of orbital object types (Payload 59%, Debris 31%, Rocket Bodies 6%, Unknown 2%). Optionally a diagram of Kessler cascade.

---

## Slide 3: Project Objectives

**Title:** What We Set Out to Build

**Key Points:**
- Build an ML regression system to predict log₁₀(collision probability) from CDM data
- Engineer domain-specific features from raw conjunction parameters
- Fuse ML predictions with physics-based collision probability calculations
- Quantify prediction uncertainty and provide confidence intervals
- Deliver explainable results via SHAP analysis
- Create an interactive Streamlit dashboard for mission operators
- Implement alerting, monitoring, and production-grade infrastructure

**Speaker Notes:**
Our objectives went well beyond just training a model. We wanted to build a complete decision-support system — one that combines machine learning with orbital mechanics, quantifies how confident it is, explains why it made each prediction, and delivers everything through an interactive operator dashboard. The goal was a system that could realistically support conjunction analysts in prioritizing which events need attention.

**Visual Suggestion:** Numbered objective list with icons — brain (ML), atom (physics), chart-bar (dashboard), bell (alerts), magnifying glass (explainability).

---

## Slide 4: Dataset — ESA Kelvins Conjunction Challenge

**Title:** Real-World Data: 162K Conjunction Data Messages

**Key Points:**
- **Source:** ESA Kelvins Collision Avoidance Challenge
- **Training set:** 162,634 CDMs | **Test set:** 24,484 CDMs
- **Raw columns:** 103 (identifiers, geometry, covariance, orbital, observation, space weather)
- **Target variable:** `risk` = log₁₀(Pc), where Pc = collision probability
- Multiple CDMs per conjunction event → time-series structure
- Risk thresholds: HIGH (> −5), MEDIUM (−7 to −5), LOW (−15 to −7), NEGLIGIBLE (< −15)

**Speaker Notes:**
We used the ESA Kelvins dataset — 162,000 real Conjunction Data Messages collected from operational screening. Each CDM describes a predicted close approach between two space objects, encoding miss distance, relative velocity, orbital elements, covariance matrices, and space weather. Importantly, multiple CDMs can belong to the same conjunction event, creating a time-series structure. The target is the log-base-10 of collision probability — so −3 means one-in-a-thousand, while −8 means negligible.

**Visual Suggestion:** Table showing dataset statistics (rows, columns, events). Example CDM record with key fields highlighted. Risk scale bar from NEGLIGIBLE to HIGH.

---

## Slide 5: System Architecture

**Title:** End-to-End Architecture

**Key Points:**
```
ESA Kelvins / Space-Track → Data Ingestion → Validation & Cleaning
  → Feature Engineering (98 raw + 46 engineered = 144 features)
    → Event-Based Splitting → Standard Scaling
      → ML Models (XGBoost, LightGBM, RF, Ridge)
      → Temporal Models (LSTM, GRU, Transformer)
      → Physics Engine (Analytic Pc, Monte Carlo)
        → Risk Fusion → Uncertainty Analysis → SHAP Explainability
          → Streamlit Dashboard + FastAPI + CLI
```
- 20+ Python modules across 17 subpackages
- Docker containerization with 3 services (API, Dashboard, Pipeline)
- SQLAlchemy persistence layer (SQLite/PostgreSQL)

**Speaker Notes:**
Here's the full architecture. Data flows from ESA's historical dataset — or potentially from Space-Track's live feed — through ingestion, validation, and cleaning. Feature engineering expands the raw 98 features into 144 by computing orbital periods, covariance ratios, temporal sequences, and uncertainty metrics. The ML pipeline trains multiple models using event-based splits to prevent data leakage. A parallel physics engine computes analytic collision probabilities. These are fused together, uncertainty is quantified, SHAP provides explainability, and everything surfaces in a Streamlit dashboard, a REST API, or a CLI tool. The whole stack is containerized with Docker Compose.

**Visual Suggestion:** Architecture flow diagram (use the one from `docs/architecture.md`). Color-coded blocks: blue for data, green for ML, orange for physics, purple for output.

---

## Slide 6: Feature Engineering — From Raw Data to Intelligence

**Title:** 144-Feature Model Matrix

**Key Points:**
- **Raw retained features:** 98 (after removing identifiers, target, leakage fields)
- **4 engineering categories:**
  1. **Static/Geometry:** relative position/velocity magnitudes, encounter duration proxy, miss-to-Mahalanobis ratio, observation utilization
  2. **Orbital:** orbital period, mean motion, perigee/apogee altitude, orbit energy, SMA/inclination differences
  3. **Covariance:** position uncertainty, sigma ratios, combined uncertainties, log covariance determinants
  4. **Temporal:** time-to-TCA transforms, CDM sequence number/fraction, observation spans
- **Leakage prevention:** explicit exclusion of `max_risk_estimate`, `max_risk_scaling`

**Speaker Notes:**
Feature engineering was critical. The raw dataset has 103 columns, but after removing identifiers and leakage-prone fields, we retain 98. We then engineer 46 additional features grounded in orbital mechanics. For example, we compute the miss-distance-to-sigma ratio — how large the predicted separation is relative to position uncertainty — which turned out to be the single most important feature. Orbital-derived features like orbital period and eccentricity proxy capture the geometry of each encounter. Temporal features capture how risk evolves as more CDMs arrive for the same event.

**Visual Suggestion:** Feature category breakdown diagram. 4 colored boxes showing the engineering categories with example features in each.

---

## Slide 7: The Most Important Feature — miss_distance_sigma_ratio

**Title:** Domain Knowledge Drives Prediction Quality

**Key Points:**
- `miss_distance_sigma_ratio = miss_distance / combined_position_uncertainty`
- Tells the model: "Is the separation large *relative to how uncertain we are?*"
- Ranked #1 by SHAP feature importance
- A small miss distance with large uncertainty → higher risk than the distance alone suggests
- Physics-informed feature engineering outperforms raw features

**Speaker Notes:**
The miss-distance-sigma-ratio deserves its own slide because it illustrates why domain knowledge matters. A 500-meter miss distance sounds safe — until you realize the position uncertainty is 300 meters. This ratio captures that relationship. SHAP analysis confirmed it as the single most important predictor. This feature wouldn't exist in a naive ML approach — it comes from understanding orbital mechanics and the structure of CDM data. Several of our best-performing engineered features follow this pattern: encoding physical relationships the model can't easily learn from raw values alone.

**Visual Suggestion:** Diagram showing two scenarios: same miss distance but different uncertainty ellipses → different risk. SHAP importance bar chart with `miss_distance_sigma_ratio` at top.

---

## Slide 8: Machine Learning Models

**Title:** Multi-Model Comparison: From Baseline to Ensemble

**Key Points:**

| Model            | Validation R² | Role                |
|------------------|--------------|---------------------|
| Ridge Regression | 0.5207       | Linear baseline     |
| Random Forest    | 0.8915       | Ensemble baseline   |
| XGBoost          | 0.9207       | Primary model       |
| LightGBM         | ~0.92        | Alternative booster |
| Stacking Ensemble| ~0.93        | Meta-learner        |

- **Event-based splitting:** 10,524 training events / 2,630 validation events / 0 overlap
- XGBoost config: 500 trees, depth 8, lr 0.05, subsample 0.8, early stopping at 50
- Stacking ensemble combines XGBoost + LightGBM + RF with Ridge meta-learner

**Speaker Notes:**
We trained four model families. Ridge regression serves as a linear baseline — R-squared of 0.52 tells us that about half the variance is explainable linearly. Random Forest jumps to 0.89, showing the importance of nonlinear feature interactions. XGBoost reaches 0.92, our primary model, trained with 500 estimators and careful regularization. LightGBM performs comparably. The stacking ensemble combines all three tree-based models using a Ridge meta-learner and pushes performance further. Critically, we use event-based splitting — no CDMs from the same conjunction appear in both training and validation — to prevent information leakage.

**Visual Suggestion:** Model comparison bar chart (R² values). Training pipeline flowchart showing the progression from baseline to ensemble.

---

## Slide 9: Temporal Deep Learning Models

**Title:** Sequence Modeling for CDM Time Series

**Key Points:**
- Conjunction events contain multiple CDMs → natural time series
- **LSTM:** Long Short-Term Memory for capturing temporal dependencies in CDM sequences
- **GRU:** Gated Recurrent Unit — lighter alternative with comparable performance
- **Transformer:** Self-attention mechanism for parallel sequence processing
- Input: CDM sequence per event → Output: risk prediction at each timestep
- Captures how risk estimates evolve as more data arrives before TCA

**Speaker Notes:**
Beyond static models, we implemented three temporal architectures. Each conjunction event generates a sequence of CDMs over days or weeks — early CDMs have high uncertainty, and as TCA approaches, the estimates sharpen. LSTM and GRU networks process these sequences step by step, learning how risk evolves. The Transformer model uses self-attention to weigh which earlier CDMs most influence the current risk estimate. These models capture a dimension that static models miss: the trajectory of risk refinement over time.

**Visual Suggestion:** CDM sequence diagram showing Event with CDMs over time. LSTM/GRU cell diagram. Attention heatmap visualization concept.

---

## Slide 10: Physics Engine

**Title:** Independent Physics Verification Layer

**Key Points:**
- **Analytic Pc calculation:** 2D probability of collision using encounter geometry + covariance
- **Covariance validation:** checks for positive-definite matrices, realistic uncertainty bounds
- **Monte Carlo simulation:** statistical collision probability via orbit sampling
- **Mahalanobis distance:** separation weighted by combined position uncertainty
- Verified on 20 validation conjunctions: **20/20 successful** non-null analytic Pc outputs

**Speaker Notes:**
The physics engine is independent of the ML models. It computes collision probability analytically using the encounter geometry and covariance information from each CDM. The analytic approach models the collision as a 2D probability integral over the combined uncertainty ellipse. Monte Carlo simulation provides a statistical cross-check by sampling perturbed orbits. The Mahalanobis distance gives a single number: how many "sigmas" apart are the objects? We validated this on 20 conjunctions from the validation set — all produced meaningful analytic Pc values that we could compare against the ML estimates.

**Visual Suggestion:** 2D encounter plane diagram showing combined covariance ellipse and hard-body radius. Monte Carlo scatter plot concept. Formula for analytic Pc.

---

## Slide 11: Risk Fusion — ML Meets Physics

**Title:** Combining ML Predictions with Physics Analysis

**Key Points:**
- **Fusion module** merges ML risk score and physics-based Pc into unified assessment
- **Confidence scoring:** quantifies reliability based on model agreement, data quality, uncertainty bounds
- **Disagreement detection:** flags cases where ML and physics significantly diverge
  - Divergence may indicate: unusual orbital geometry, poor CDM data quality, or model limitation
- Final fused risk = weighted combination accounting for confidence and agreement

**Speaker Notes:**
Neither ML nor physics alone is sufficient. ML excels at pattern recognition across 144 features but can be fooled by out-of-distribution data. Physics is principled but relies on idealized assumptions. Our fusion module combines both: when they agree, confidence is high. When they disagree, the system flags the conjunction for manual review and adjusts the confidence score downward. The disagreement detector itself becomes a safety mechanism — it catches cases where the ML model may be extrapolating beyond its training distribution.

**Visual Suggestion:** Venn diagram: ML circle + Physics circle → Fusion overlap. Decision tree: Agree → high confidence, Disagree → flag for review. Confidence meter graphic.

---

## Slide 12: SHAP Explainability

**Title:** Why Did the Model Predict High Risk?

**Key Points:**
- SHAP (SHapley Additive exPlanations) decomposes each prediction into per-feature contributions
- Computed on 200 validation samples
- Top contributing features:
  1. `miss_distance_sigma_ratio`
  2. `miss_distance`
  3. `mahalanobis_distance`
  4. `combined_position_uncertainty`
  5. Relative velocity components
- Natural language explanation module translates SHAP values into human-readable sentences

**Speaker Notes:**
Black-box predictions aren't acceptable in safety-critical applications. SHAP gives us a principled decomposition: for each prediction, which features pushed the risk estimate up, and which pushed it down? For example, a high-risk prediction might be explained as: "miss distance sigma ratio is low — the objects are close relative to the uncertainty — and the relative velocity is high, reducing the time window for maneuver." Our explainability module even translates these SHAP values into natural language sentences that a non-ML operator could understand.

**Visual Suggestion:** SHAP waterfall plot for one conjunction. SHAP summary beeswarm plot showing feature importance distribution. Example natural language explanation box.

---

## Slide 13: Uncertainty Quantification

**Title:** Not Just a Prediction — How Confident Are We?

**Key Points:**
- **Prediction intervals:** bounds around each risk estimate (e.g., 90% CI)
- **Calibration analysis:** do 90% intervals actually contain 90% of true values?
- **Confidence scoring:** integrated with risk fusion module
- Critical for operational decisions: "predicted risk is −4.8 (HIGH), but uncertainty spans −6.1 to −3.5"
- Avoids false precision — a single point estimate can mislead

**Speaker Notes:**
A prediction of log-Pc equals minus 4.8 looks precise, but how much should an operator trust it? Our uncertainty module computes prediction intervals — instead of a single number, we give a range. If the 90% interval spans from minus 6 to minus 3.5, that tells the operator the prediction straddles the HIGH/MEDIUM boundary, and they should be cautious. We calibrate these intervals to ensure they're statistically reliable. This is essential for any safety-critical system — operators need to know not just the prediction but how reliable it is.

**Visual Suggestion:** Risk prediction with confidence interval bands. Calibration plot (predicted coverage vs. actual coverage). Example output showing point estimate + interval.

---

## Slide 14: Interactive Dashboard

**Title:** 13-Page Streamlit Dashboard with Glassmorphism Design

**Key Points:**
- **Dark glassmorphism theme** — modern, professional operator interface
- **Core pages:** Mission Overview, Conjunction Monitor, CDM Details, AI Risk Analysis, Physics Analysis, SHAP Explanations, Uncertainty Analysis, Temporal/Event View, 3D Orbital Visualization
- **Advanced pages:** Ensemble Comparison, Alert Dashboard, Report Generator, System Status
- **Orbital Tracker:** animated conjunction visualization with real-time orbital motion
- Interactive Plotly charts: zoom, hover, filter, drill-down
- All pages connected to the same backend prediction engine

**Speaker Notes:**
The dashboard is the primary operator interface, built with Streamlit and styled with a dark glassmorphism theme — semi-transparent cards with blur effects that give it a modern, professional feel. It has 13 pages covering every aspect of the system: from high-level mission overview and conjunction monitoring, through AI risk analysis with SHAP explanations, to a 3D orbital tracker that animates conjunction geometry. The ensemble comparison page lets analysts compare models side-by-side. The alert dashboard shows triggered notifications. Every chart is interactive via Plotly — operators can zoom, hover for details, and drill down.

**Visual Suggestion:** Dashboard screenshots — Mission Overview page, AI Risk Analysis page, 3D Orbital Tracker, Alert Dashboard. Tiled layout of 4 screenshots.

---

## Slide 15: REST API and CLI

**Title:** Production Interfaces: FastAPI + Click CLI

**Key Points:**
- **FastAPI REST API** (port 8000):
  - `POST /predict` — single CDM risk prediction
  - `POST /batch` — batch prediction
  - `GET /health` — health check
  - CDM management endpoints
  - Prediction history endpoints
  - Auto-generated OpenAPI/Swagger documentation
- **Click CLI tool** (`orbital-sentinel`):
  - `predict` — run predictions from command line
  - `train` — trigger model training
  - `evaluate` — run evaluation suite
  - `status` — system health check
- **SQLAlchemy persistence:** SQLite (development) / PostgreSQL (production)

**Speaker Notes:**
Beyond the dashboard, we built two additional interfaces for different use cases. The FastAPI REST API exposes prediction as a service — send CDM data as JSON, receive a risk assessment with confidence and SHAP explanations. It includes batch processing for bulk screening. The CLI tool provides command-line access for pipeline operations, useful for scripting and automation. Both share the same underlying inference engine and database layer, which uses SQLAlchemy and supports SQLite for development or PostgreSQL for production deployment.

**Visual Suggestion:** API endpoint table. Swagger UI screenshot. CLI help output screenshot. Architecture showing API + CLI + Dashboard all connecting to the same backend.

---

## Slide 16: Monitoring, Alerts, and Data Quality

**Title:** Operational Monitoring and Automated Alerts

**Key Points:**
- **Data Quality Monitoring:**
  - Completeness checks — missing value rates per feature
  - Distribution validation — detect anomalous CDM batches
  - Schema enforcement at ingestion boundary
- **Drift Detection:** flag when incoming data distribution shifts from training data
- **Alert System:**
  - Configurable rules: threshold-based, rate-of-change, multi-condition
  - Alert manager with prioritization and deduplication
  - Notification channels: webhook integration + structured logging
- System Status page in dashboard for operational overview

**Speaker Notes:**
A deployed ML system needs monitoring. Our data quality module validates incoming CDMs against expected schemas and distributions — catching issues like missing covariance fields or anomalous miss distances before they corrupt predictions. The drift detector compares incoming feature distributions against the training data baseline. The alert system supports configurable rules: "trigger HIGH alert if predicted risk exceeds minus 5 and confidence exceeds 0.8." Alerts are managed with prioritization and deduplication, and can fire webhooks for integration with external notification systems like Slack or PagerDuty.

**Visual Suggestion:** Monitoring flow diagram. Alert rule configuration example. System Status dashboard page screenshot.

---

## Slide 17: Testing, Quality, and DevOps

**Title:** Production-Grade Engineering Practices

**Key Points:**
- **Test suite:** 16 test modules | 81 tests passed | 0 failures
  - Unit tests: individual module verification
  - Integration tests: pipeline end-to-end validation
  - Covers: ingestion, preprocessing, features, models, physics, fusion, evaluation, API, alerts, database, reporting, uncertainty, validation, explainability
- **Code quality:** Pylint score **~9.69/10**
- **CI/CD:** GitHub Actions pipeline — lint, test, build on every push
- **Docker:** 3-service Compose stack (API + Dashboard + Pipeline) with health checks
- **Configuration:** YAML-based, environment variables via `.env`

**Speaker Notes:**
Engineering quality was a priority. We have 16 test modules covering every subsystem — from data ingestion through to the REST API. All 81 tests pass. Pylint scores 9.69 out of 10, reflecting clean, consistent code. GitHub Actions runs linting and tests on every push, catching regressions immediately. The Docker Compose configuration defines three services — the API, the dashboard, and a training pipeline — with health checks, network isolation, and volume mounts for data persistence. The entire stack can be launched with one command.

**Visual Suggestion:** Test results summary table. Pylint score badge. CI/CD pipeline diagram. Docker Compose architecture showing the 3 services.

---

## Slide 18: Results Summary

**Title:** Quantitative Performance Results

**Key Points:**

| Metric                | Value        |
|-----------------------|-------------|
| Training CDMs         | 162,634     |
| Features              | 144         |
| Training Events       | 10,524      |
| Validation Events     | 2,630       |
| Event Overlap         | 0           |
| XGBoost R²            | 0.9207      |
| Random Forest R²      | 0.8915      |
| Ridge R²              | 0.5207      |
| High-Risk Precision   | ~0.698      |
| High-Risk Recall      | ~0.423      |
| High-Risk ROC-AUC     | ~0.967      |
| Physics Verification  | 20/20       |
| Automated Tests       | 81 passed   |
| Pylint Score          | ~9.69/10    |

**Speaker Notes:**
Here are the headline numbers. XGBoost explains 92% of variance in collision probability — a strong result considering the complexity and noise in real conjunction data. The jump from Ridge's 52% to XGBoost's 92% demonstrates that the nonlinear relationships in encounter geometry are critical. For binary high-risk classification at the −5 threshold, ROC-AUC reaches 0.967, meaning the model reliably separates dangerous conjunctions from benign ones. Precision at 0.70 means 70% of flagged events truly are high-risk, reducing analyst workload on false alarms. All 20 physics verification runs produced valid results, and the software passes all 81 automated tests.

**Visual Suggestion:** Results table prominently displayed. Bar chart comparing R² across models. ROC curve plot. Confusion matrix for high-risk threshold.

---

## Slide 19: Key Innovations and Technical Contributions

**Title:** What Makes Orbital Sentinel Different

**Key Points:**
1. **Physics-ML Fusion:** independent verification layer catches ML blind spots; disagreement detection as a safety mechanism
2. **Domain-Driven Feature Engineering:** 46 features grounded in orbital mechanics (miss_distance_sigma_ratio, orbital period, covariance ratios) outperform raw data
3. **Event-Based Validation:** prevents data leakage across CDMs from the same conjunction event
4. **Temporal Deep Learning:** LSTM/GRU/Transformer models capture risk evolution across CDM sequences
5. **Explainability-First Design:** every prediction comes with SHAP decomposition and natural language explanation
6. **Uncertainty-Aware Predictions:** confidence intervals prevent false precision in safety-critical decisions
7. **Full Production Stack:** API + CLI + Dashboard + Docker + CI/CD + Monitoring + Alerts

**Speaker Notes:**
Several aspects distinguish this system. The physics-ML fusion is not just an ensemble — it's an independent cross-check that flags disagreements as safety signals. Our feature engineering is domain-driven, not automated — each engineered feature has a physical interpretation. Event-based splitting is a methodological contribution that prevents a subtle but common form of leakage in CDM data. The temporal models address a dimension that most static approaches ignore: how risk evolves over an event's lifetime. And the full production stack — API, CLI, dashboard, monitoring, alerts, Docker — makes this more than a Jupyter notebook experiment.

**Visual Suggestion:** Innovation highlight cards — one per point with a small icon and 1-line description. Star or lightbulb icons.

---

## Slide 20: Technology Stack

**Title:** Technologies and Frameworks

**Key Points:**

| Layer              | Technology                                |
|--------------------|------------------------------------------|
| Core Language      | Python 3.11                              |
| ML Framework       | XGBoost, LightGBM, Scikit-learn          |
| Deep Learning      | PyTorch (LSTM, GRU, Transformer)         |
| Data Processing    | Pandas, NumPy                            |
| Explainability     | SHAP                                     |
| Visualization      | Plotly, Streamlit                        |
| API                | FastAPI, Uvicorn                         |
| CLI                | Click                                    |
| Database           | SQLAlchemy (SQLite / PostgreSQL)         |
| Testing            | Pytest, Pylint                           |
| DevOps             | Docker, Docker Compose, GitHub Actions   |
| Data Source         | ESA Kelvins, Space-Track API            |
| Configuration      | YAML, python-dotenv                      |

**Speaker Notes:**
The stack spans the full ML engineering spectrum. Python with XGBoost and LightGBM for the core ML. PyTorch for the temporal deep learning models. Pandas and NumPy for data processing. SHAP for explainability. Plotly and Streamlit for the interactive dashboard. FastAPI for the REST interface. SQLAlchemy for persistence. Docker and GitHub Actions for DevOps. Each technology was chosen for its specific strengths in this domain.

**Visual Suggestion:** Technology stack diagram — layered blocks from data at the bottom to user interface at the top, with logos for each technology.

---

## Slide 21: Live Demo Walkthrough

**Title:** System Demonstration

**Key Points (Demo Script):**
1. **Launch:** `docker-compose up -d` or `streamlit run dashboard/app.py`
2. **Mission Overview:** show event count, risk distribution, key metrics
3. **Conjunction Monitor:** browse conjunction list, filter by risk level
4. **AI Risk Analysis:** select an event → show prediction, confidence, risk category
5. **SHAP Explanations:** show feature contributions for the selected event
6. **3D Orbital Tracker:** animate conjunction geometry in 3D
7. **Physics Analysis:** show analytic Pc alongside ML prediction
8. **Alert Dashboard:** demonstrate triggered alerts
9. **System Status:** show monitoring metrics, data quality
10. **API:** (optional) `curl` a prediction endpoint

**Speaker Notes:**
Let me walk through a live demo. We start by launching the system — one command brings up the API and dashboard. The Mission Overview page gives a bird's-eye view of all tracked conjunctions. Clicking into the Conjunction Monitor, we can filter by risk level to find high-priority events. Selecting one event, the AI Risk Analysis page shows the prediction with confidence bounds. The SHAP page breaks down exactly which features drove this prediction. The 3D Orbital Tracker visualizes the encounter geometry with animated orbital paths. We can cross-reference with the Physics Analysis page. The Alert Dashboard shows any triggered notifications. And the System Status page confirms data quality and model health.

**Visual Suggestion:** Live demo on screen. Have backup screenshots ready in case of technical issues.

---

## Slide 22: Future Directions

**Title:** Roadmap and Potential Extensions

**Key Points:**
- **Real-Time Space-Track Integration:** continuous monitoring of live CDM feeds
- **Maneuver Recommendation Engine:** suggest collision-avoidance burn parameters
- **Physics-Informed Neural Networks (PINNs):** embed orbital mechanics constraints into the loss function
- **Multi-Constellation Analysis:** scale to mega-constellation monitoring (Starlink, OneWeb)
- **Federated Learning:** collaborative model training across space agencies without sharing raw data
- **Automated Sensitivity Analysis:** what-if scenarios for miss distance, velocity, uncertainty changes
- **High-Fidelity Propagation:** J2+ perturbation models for improved physics accuracy

**Speaker Notes:**
Several directions could extend this work. The most immediate is real-time Space-Track integration for continuous monitoring. A maneuver recommendation engine could suggest optimal avoidance burns. Physics-informed neural networks could embed Kepler's laws directly into the model's loss function, improving generalization. As mega-constellations like Starlink grow, scaling to thousands of simultaneous conjunctions becomes critical. Federated learning could enable international collaboration without sharing sensitive orbital data. Each of these builds directly on the architecture we've implemented.

**Visual Suggestion:** Roadmap timeline graphic. Icons for each future direction branching from the current system.

---

## Slide 23: Summary and Conclusion

**Title:** Orbital Sentinel — Summary

**Key Points:**
- Built a **complete ML pipeline** for satellite conjunction risk assessment
- Achieved **R² = 0.92** and **ROC-AUC = 0.967** on real ESA data with zero event leakage
- **Physics-ML fusion** provides independent verification and safety mechanisms
- **Explainability (SHAP)** and **uncertainty quantification** make predictions trustworthy
- **144 domain-engineered features** demonstrate the value of physics-informed ML
- **Production-grade system:** 13-page dashboard, REST API, CLI, Docker, CI/CD, 81 tests
- Design principle: *"ML for pattern recognition, physics for verification, uncertainty for honesty, explainability for trust"*

**Speaker Notes:**
To summarize: Orbital Sentinel demonstrates that satellite conjunction risk assessment can be automated effectively with machine learning, but only when combined with domain expertise, physics verification, uncertainty awareness, and explainability. The 0.92 R-squared and 0.967 AUC on real ESA data, achieved with rigorous event-based validation, show that the approach works on realistic conjunction screening problems. But the numbers alone don't capture the full contribution — it's the complete system: the physics cross-check, the confidence intervals, the SHAP explanations, the operator dashboard, and the production infrastructure that make this a practical decision-support tool rather than just a regression model.

**Visual Suggestion:** Summary card with the 4 pillars: ML + Physics + Uncertainty + Explainability. Key metrics highlighted. System screenshot collage in background.

---

## Slide 24: Thank You and Q&A

**Title:** Questions?

**Key Points:**
- Thank you for your attention
- Repository: [GitHub URL]
- Technologies: Python | XGBoost | SHAP | FastAPI | Streamlit | Docker
- Contact: [Your Email]

**Speaker Notes:**
Thank you. I'm happy to take any questions — whether about the ML approach, the physics engine, the feature engineering decisions, the dashboard design, or any other aspect of the system. I can also demonstrate any specific feature of the system live if that would be helpful.

**Visual Suggestion:** Clean "Thank You" slide with contact information and a QR code linking to the repository.

---

## Appendix: Backup Slides

### Backup A: Detailed Feature List (144 features)

- 98 raw retained features (see README Section 12 for full list)
- 46 engineered features across 4 categories (see README Sections 16–20)

### Backup B: XGBoost Hyperparameters

| Parameter           | Value |
|---------------------|-------|
| n_estimators        | 500   |
| max_depth           | 8     |
| learning_rate       | 0.05  |
| subsample           | 0.8   |
| colsample_bytree    | 0.8   |
| random_state        | 42    |
| early_stopping      | 50    |

### Backup C: Event-Based Split Details

- 10,524 training events → ~130K CDMs
- 2,630 validation events → ~32K CDMs
- 0 overlapping events between splits
- Split by `event_id`, not by row

### Backup D: API Endpoint Reference

| Method | Endpoint      | Description                 |
|--------|---------------|-----------------------------|
| POST   | /predict      | Single CDM risk prediction  |
| POST   | /batch        | Batch risk prediction       |
| GET    | /health       | Service health check        |
| GET    | /models       | List available models       |
| GET    | /predictions  | Prediction history          |
