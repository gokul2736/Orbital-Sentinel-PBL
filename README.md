# ORBITAL SENTINEL

## Automated Machine Learning Triage System for Space Debris Conjunction Analysis

**Project Type:** AI/ML + Physics-Based Space Safety Analysis
**Primary Dataset:** ESA Kelvins Collision Avoidance Challenge Dataset
**Primary Model:** XGBoost Regression
**Interface:** Streamlit + Plotly
**Explainability:** SHAP
**Validation:** Event-Based Validation + Physics Verification + Automated Testing

---

# 1. Project Overview

Orbital Sentinel is an automated conjunction-risk assessment system designed to analyse close approaches between space objects.

The system combines:

1. Historical conjunction data
2. Data preprocessing
3. Domain-specific feature engineering
4. Machine learning
5. Physics-based verification
6. Uncertainty estimation
7. Explainable AI
8. Interactive visualization

The primary machine-learning task is **regression**.

The model predicts:

`log10(Pc)`

where:

* `Pc` = collision probability
* `log10(Pc)` = base-10 logarithm of collision probability

A less-negative value represents a higher collision probability.

Example:

```text
Pc = 10^-6
log10(Pc) = -6
```

The ESA dataset defines `risk` as the base-10 logarithm of collision probability.

---

# 2. What Is a Conjunction?

A conjunction is a predicted close approach between two space objects.

A conjunction does NOT automatically mean a collision.

The system evaluates the encounter using information such as:

* Miss distance
* Relative velocity
* Time to closest approach
* Relative position
* Relative velocity components
* Covariance and uncertainty
* Orbital parameters
* Observation information
* Object characteristics

The objective is to estimate and analyse the level of collision risk.

---

# 3. What Is TCA?

TCA means:

**Time of Closest Approach**

It is the predicted time when the two objects reach their minimum separation.

Important conjunction quantities are generally evaluated around TCA:

```text
Object 1
     \
      \
       \       Closest Approach
        \          ●
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

---

# 4. Dataset

Orbital Sentinel uses the ESA Kelvins Collision Avoidance Challenge dataset.

Dataset statistics:

```text
Training rows: 162,634
Testing rows: 24,484
Raw columns: 103
```

Each row corresponds to a CDM.

Multiple CDMs can belong to the same conjunction event.

Therefore:

```text
Event
 ├── CDM 1
 ├── CDM 2
 ├── CDM 3
 ├── CDM 4
 └── ...
```

An event can therefore be considered a time series of CDMs.

---

# 5. Raw Dataset Columns

The original ESA dataset contains 103 columns.

These can be grouped into the following categories.

## 5.1 Event and Target Information

* `event_id`
* `mission_id`
* `risk`
* `time_to_tca`

`event_id` identifies a close-approach event.

`mission_id` identifies the affected mission.

`risk` is the target variable.

`time_to_tca` represents the time between CDM creation and TCA.

---

# 6. Relative Geometry Features

These describe the physical encounter between the two objects.

Important raw variables include:

* `miss_distance`
* `relative_speed`
* `relative_position_r`
* `relative_position_t`
* `relative_position_n`
* `relative_velocity_r`
* `relative_velocity_t`
* `relative_velocity_n`
* `azimuth`
* `elevation`
* `geocentric_latitude`

### Meaning

`miss_distance`:

Predicted separation between the two objects at TCA.

`relative_speed`:

Speed of one object relative to the other at TCA.

Relative position and velocity are represented in three directions:

* R = radial
* T = transverse / along-track
* N = normal / cross-track

---

# 7. Object Information

The dataset contains information about the object involved in the conjunction.

Important fields include:

* `c_object_type`
* object-related physical characteristics
* radar cross-section
* area-to-mass information
* object size estimates

`c_object_type` describes the type of object involved in the potential collision.

---

# 8. Covariance and Uncertainty Features

Covariance features describe uncertainty in the estimated state of each object.

For both target (`t`) and chaser (`c`) objects, the dataset contains variables such as:

* `x_sigma_r`
* `x_sigma_t`
* `x_sigma_n`
* `x_sigma_rdot`
* `x_sigma_tdot`
* `x_sigma_ndot`

and covariance/correlation terms such as:

* `x_cn_r`
* `x_cn_t`
* `x_cndot_n`
* `x_cndot_r`
* `x_cndot_rdot`
* `x_cndot_t`
* `x_cndot_tdot`
* `x_ct_r`
* `x_ctdot_n`
* `x_ctdot_r`
* `x_ctdot_rdot`
* `x_ctdot_t`
* `x_ctdot_tdot`
* `x_crdot_n`
* `x_crdot_t`
* `x_crdot_r`
* `x_position_covariance_det`

Here `x` means the feature exists for both:

```text
t = target
c = chaser
```

---

# 9. Orbital Features

Orbital-state information includes:

* `x_h_apo`
* `x_h_per`
* `x_ecc`
* `x_j2k_inc`
* `x_j2k_sma`

These represent orbital properties such as:

* Apogee
* Perigee
* Eccentricity
* Inclination
* Semi-major axis

They describe the orbital configuration of the objects.

---

# 10. Observation / Orbit Determination Features

The dataset also contains information describing how the object's orbit was determined.

Examples:

* `x_actual_od_span`
* `x_obs_available`
* `x_obs_used`
* `x_recommended_od_span`
* `x_residuals_accepted`
* `x_time_lastob_end`
* `x_time_lastob_start`
* `x_weighted_rms`

These provide information about the quantity and quality of observations used in orbit determination.

---

# 11. Space Weather Features

The dataset contains environmental variables:

* `F10`
* `F3M`
* `SSN`
* `AP`

These represent space-weather / solar-activity information that can affect the space environment and orbital dynamics.

---

# 12. Why Are There 98 Raw Model Features?

The original dataset has:

```text
103 columns
```

However, not every column should be supplied directly to the ML model.

The following five are excluded:

```text
event_id
mission_id
risk
max_risk_estimate
max_risk_scaling
```

Therefore:

```text
103
- 2 identifiers
- 1 target
- 2 leakage-related fields
---------------------------
= 98 retained raw features
```

The identifiers are used for grouping and tracking, but are not treated as predictive numerical inputs.

`risk` is the value the model is trying to predict.

The maximum-risk fields are excluded because they can introduce target-related information into the model.

---

# 13. What Is a Feature?

A feature is an input variable used by the machine-learning model.

Example:

```text
miss_distance
relative_speed
time_to_tca
t_sigma_r
c_sigma_r
orbital parameters
```

The model uses these inputs to learn the relationship between conjunction characteristics and collision risk.

---

# 14. What Is Feature Engineering?

Feature engineering means creating additional useful variables from existing information.

Instead of giving the model only:

```text
miss_distance
uncertainty
relative velocity
```

we can calculate:

```text
miss distance relative to uncertainty
relative position magnitude
relative velocity magnitude
orbital period
mean motion
observation utilization
```

These derived quantities can make physical relationships easier for the model to learn.

---

# 15. The 46 Engineered Features

The final feature matrix contains:

```text
98 raw features
+
46 engineered/derived features
=
144 model features
```

The engineered features are derived from four main groups.

---

# 16. Static / Geometry-Derived Features

The implementation derives quantities including:

```text
relative_position_magnitude
relative_velocity_magnitude
encounter_duration_proxy
t_obs_utilization
c_obs_utilization
t_od_span_ratio
c_od_span_ratio
miss_to_mahalanobis_ratio
```

### relative_position_magnitude

Computed from:

```text
sqrt(R² + T² + N²)
```

It represents the magnitude of the relative-position vector.

### relative_velocity_magnitude

Computed from the three relative velocity components.

### encounter_duration_proxy

Conceptually:

```text
miss_distance / relative_speed
```

This provides an approximate encounter-duration scale.

### observation utilization

Conceptually:

```text
observations_used / observations_available
```

It describes how much of the available observation information was actually used.

### OD span ratio

Compares:

```text
actual orbit-determination span
/
recommended orbit-determination span
```

---

# 17. Orbital-Derived Features

The orbital feature module derives:

```text
t_orbital_period
t_mean_motion
t_perigee_alt
t_apogee_alt
t_orbit_energy

c_orbital_period
c_mean_motion
c_perigee_alt
c_apogee_alt
c_orbit_energy

sma_difference
inclination_difference
t_orbit_eccentricity_proxy
c_orbit_eccentricity_proxy
```

### Orbital period

Derived from semi-major axis using the orbital-period relationship.

### Mean motion

Represents orbital angular rate.

### Perigee / apogee altitude

Derived from:

```text
semi-major axis
eccentricity
Earth radius
```

### Semi-major-axis difference

Measures the absolute difference between target and chaser semi-major axes.

### Inclination difference

Measures the absolute difference between their orbital inclinations.

---

# 18. Covariance-Derived Features

The covariance module derives uncertainty-related features including:

```text
t_position_uncertainty
t_sigma_max
t_sigma_min
t_sigma_ratio

c_position_uncertainty
c_sigma_max
c_sigma_min
c_sigma_ratio

combined_position_uncertainty
combined_sigma_r
combined_sigma_t
combined_sigma_n

miss_distance_sigma_ratio

t_log_cov_det
c_log_cov_det
```

These features transform multiple covariance values into easier-to-use quantities.

---

# 19. Important Feature: miss_distance_sigma_ratio

One of the most important engineered features is:

```text
miss_distance_sigma_ratio
```

Conceptually:

```text
miss distance
-----------------------------
combined position uncertainty
```

This tells the model how large the predicted separation is relative to the uncertainty scale.

It is more informative than considering distance alone.

This feature was also the top reported SHAP feature in the completed explainability analysis.

---

# 20. Temporal Features

The temporal module derives:

```text
time_to_tca_hours
time_to_tca_squared
log_time_to_tca
is_close_approach

cdm_sequence_number
cdm_total_in_event
cdm_sequence_fraction

t_observation_span
c_observation_span
```

These describe how the conjunction evolves over time.

For example:

```text
CDM 1
CDM 2
CDM 3
CDM 4
```

can be represented using sequence-related variables.

---

# 21. Mahalanobis Distance

Mahalanobis distance measures distance relative to an uncertainty distribution.

Conceptually:

```text
physical separation
+
covariance information
        ↓
Mahalanobis distance
```

Formula:

```text
D = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
```

where:

* `x` = observed/predicted state
* `μ` = reference state
* `Σ` = covariance matrix

This is useful when ordinary Euclidean distance does not adequately represent uncertainty.

---

# 22. Final Feature Matrix

The final model input is:

```text
98 retained raw features
+
46 engineered features
=
144 features
```

Therefore the model does not simply consume the original 103 columns.

The process is:

```text
103 raw dataset columns
          ↓
remove identifiers / target / leakage fields
          ↓
98 raw model features
          ↓
feature engineering
          ↓
46 derived features
          ↓
144-dimensional model matrix
```

---

# 23. Data Cleaning

The preprocessing pipeline performs:

1. Infinite-value handling
2. Missing-value handling
3. Removal of all-NaN columns
4. Categorical encoding
5. Duplicate removal
6. Median imputation for remaining numerical missing values

The categorical object-type field is encoded numerically.

---

# 24. Data Normalization

The project uses `StandardScaler`.

Conceptually:

```text
z = (x - mean) / standard deviation
```

The scaler is fitted using training data and then applied to validation data.

The trained scaler is saved as:

```text
models_saved/scaler.joblib
```

---

# 25. Preventing Data Leakage

Data leakage occurs when information unavailable at prediction time is accidentally supplied to the model.

Orbital Sentinel performs an explicit leakage check.

The preprocessing configuration excludes:

```text
event_id
mission_id
risk
max_risk_estimate
max_risk_scaling
```

This prevents identifiers and target-related information from being used as predictive inputs.

---

# 26. Event-Based Train/Validation Split

A random row split is not ideal because one event can contain multiple CDMs.

For example:

```text
Event 1001

CDM 1
CDM 2
CDM 3
CDM 4
CDM 5
```

Putting some rows into training and other rows into validation could allow the model to see almost the same physical event in both sets.

Therefore the project uses:

**event-based splitting**

Verified split:

```text
Training events:   10,524
Validation events:  2,630
Event overlap:     0
```

This provides a cleaner validation experiment.

---

# 27. Machine Learning Task

The primary ML problem is:

**Regression**

Input:

```text
144 features
```

Target:

```text
risk = log10(Pc)
```

Output:

```text
predicted log10(Pc)
```

This is not ordinary binary classification.

---

# 28. Models Used

The project contains multiple model implementations.

## Ridge Regression

Used as a simple linear baseline.

Purpose:

* establish a basic reference
* measure how much nonlinear modelling improves performance

Validation:

```text
R² = 0.5207
```

---

## Random Forest

An ensemble of decision trees.

It can model nonlinear relationships and feature interactions.

Validation:

```text
R² = 0.8915
```

---

## XGBoost

The primary completed model.

XGBoost uses gradient-boosted decision trees.

It is suitable for structured/tabular data containing nonlinear relationships and feature interactions.

Validation:

```text
R² ≈ 0.9207
```

Configuration includes:

```text
n_estimators = 500
max_depth = 8
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
early_stopping_rounds = 50
```

The trained model reached:

```text
best_iteration = 499
```

---

# 29. Why XGBoost?

Collision-risk data contains nonlinear relationships.

For example:

```text
miss distance
        +
relative velocity
        +
uncertainty
        +
orbital geometry
```

may interact in ways that cannot be represented well by a simple linear model.

XGBoost can learn nonlinear decision structures and interactions between tabular features.

Therefore it was selected as the primary completed model.

---

# 30. ML Training Process

The training workflow is:

```text
ESA Dataset
     ↓
Schema Validation
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Leakage Check
     ↓
Event-Based Split
     ↓
Training Features / Target
     ↓
StandardScaler
     ↓
Model Training
     ↓
Validation Prediction
     ↓
Metric Calculation
```

The pipeline trains the baseline model set and compares their validation performance.

---

# 31. XGBoost Training

The model receives:

```text
X_train
```

containing the 144 features.

The target is:

```text
y_train = log10(Pc)
```

Validation data:

```text
X_val
y_val
```

is kept separate.

XGBoost is trained on the training split and evaluated using the event-isolated validation split.

Early stopping is supported using the validation set.

The final model is saved using Joblib.

---

# 32. Model Evaluation

Regression metrics include:

### MAE

Mean Absolute Error.

Measures the average absolute prediction error.

### RMSE

Root Mean Squared Error.

Penalizes large errors more strongly than MAE.

### R²

Coefficient of determination.

Measures how much variance in the target is explained by the model relative to a baseline.

---

# 33. Model Results

| Model         | Validation R² |
| ------------- | ------------: |
| Ridge         |        0.5207 |
| Random Forest |        0.8915 |
| XGBoost       |      ≈ 0.9207 |

XGBoost was the strongest completed primary model in the validation experiment.

---

# 34. Binary Risk Evaluation

Although the primary problem is regression, the predictions can also be converted into a high-risk classification for analysis.

The project uses:

```text
log10(Pc) >= -5.0
```

as an evaluation threshold.

That corresponds to:

```text
Pc >= 10^-5
```

Reported metrics:

```text
Precision ≈ 0.698
Recall    ≈ 0.423
F1        ≈ 0.527
ROC-AUC   ≈ 0.967
```

This threshold is an evaluation configuration and should not automatically be interpreted as an operational maneuver threshold.

---

# 35. SHAP Explainability

Machine-learning predictions should not be treated as unexplained numbers.

SHAP is used to determine which features contributed to a prediction.

Conceptually:

```text
144 features
      ↓
XGBoost
      ↓
prediction
      ↓
SHAP
      ↓
feature contributions
```

The completed pipeline calculated SHAP values on a sample of:

```text
200 validation samples
```

The top reported feature was:

```text
miss_distance_sigma_ratio
```

---

# 36. Physics Engine

The ML model is supported by physics-based analysis.

The physics layer includes:

* collision-probability calculations
* relative geometry
* covariance information
* Mahalanobis analysis
* orbital checks
* analytic Pc calculations
* physics verification

The purpose is not to replace the ML model.

It provides an independent physics-oriented analysis layer.

---

# 37. ML + Physics Architecture

```text
                CONJUNCTION DATA
                       ↓
              PREPROCESSING
                       ↓
              144 FEATURES
                       ↓
              ┌───────────────┐
              │    XGBoost    │
              └───────┬───────┘
                      ↓
                ML Risk Estimate
                      │
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
   Physics Engine           Uncertainty
          ↓                       ↓
   Analytic Pc             Confidence /
          │                 Intervals
          └───────────┬───────────┘
                      ↓
                 Risk Fusion
                      ↓
                 SHAP Analysis
                      ↓
             Streamlit Dashboard
```

---

# 38. Physics Verification

The completed pipeline selects 20 validation conjunctions and runs them through the physics verification workflow.

Result:

```text
20 / 20
```

successfully completed with non-null analytic Pc outputs.

This is an execution-validation result, not a claim of 100% physics accuracy.

---

# 39. Uncertainty Analysis

The project includes prediction-uncertainty analysis.

Instead of displaying only:

```text
Prediction = -6.2
```

the system can also estimate an interval around the prediction.

This allows the system to communicate:

```text
prediction
+
uncertainty
```

rather than presenting a single number as absolute truth.

---

# 40. Temporal Analysis

Each conjunction can contain multiple CDMs.

Example:

```text
Event 501

CDM 1 → early prediction
CDM 2 → updated prediction
CDM 3 → updated prediction
CDM 4 → final available information
```

Therefore risk can evolve over time.

The project includes temporal feature engineering and modules for:

* event tracking
* sequencing
* temporal feature extraction
* LSTM
* GRU
* Transformer

The primary completed and evaluated model in the main pipeline is XGBoost.

---

# 41. Space-Track Integration

The project contains integration scripts for Space-Track.

The intended architecture is:

```text
Space-Track
     ↓
Authentication
     ↓
CDM Retrieval
     ↓
CDM Parsing
     ↓
Feature Extraction
     ↓
ML + Physics
     ↓
Risk Assessment
     ↓
Dashboard
```

Live operation depends on external API authentication and availability.

---

# 42. Dashboard

The Streamlit dashboard connects the backend analysis with an interactive user interface.

Main visualization technology:

**Plotly**

Plotly is used for interactive:

* risk graphs
* event timelines
* feature plots
* 2D visualizations
* 3D orbital/conjunction visualizations

Streamlit provides the application interface.

---

# 43. What the Dashboard Provides

The dashboard is designed around:

1. Mission overview
2. Conjunction monitoring
3. Event details
4. CDM information
5. AI risk
6. Physics analysis
7. Uncertainty
8. SHAP explanations
9. Temporal/event analysis
10. Model analysis
11. Data quality/system status

---

# 44. What-If / Sensitivity Analysis

A future/advanced capability is sensitivity analysis.

Instead of asking only:

```text
What is the current predicted risk?
```

the system can ask:

```text
What happens if miss distance changes?
What happens if relative velocity changes?
What happens if uncertainty increases?
What happens if uncertainty decreases?
```

This produces scenario-based analysis.

These are counterfactual scenarios, not actual observations.

---

# 45. Automated Pipeline

The complete pipeline contains 11 major stages:

```text
1. Load ESA dataset
2. Validate schema and quality
3. Check target leakage
4. Preprocess and engineer features
5. Train baseline models
6. Evaluate best model
7. Estimate uncertainty
8. Run physics verification
9. Fuse risk assessments
10. Generate SHAP explanations
11. Save model/results/artifacts
```

The pipeline is implemented in:

```text
scripts/run_pipeline.py
```

---

# 46. Saved Model Artifacts

The training pipeline saves:

```text
models_saved/
├── XGBoost_model.joblib
├── scaler.joblib
└── feature_names.json
```

It also saves pipeline results under:

```text
proofs/pipeline_results.json
```

These artifacts allow the dashboard/inference layer to use the trained model without retraining every time.

---

# 47. Software Testing

The project includes automated tests.

Verified test result:

```text
81 passed
0 failed
```

Static code quality:

```text
Pylint ≈ 9.69 / 10
```

This validates the software implementation in addition to the ML metrics.

---

# 48. Technology Stack

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Core implementation                |
| Pandas       | Data processing                    |
| NumPy        | Numerical computation              |
| Scikit-learn | Preprocessing, scaling and metrics |
| XGBoost      | Primary ML model                   |
| LightGBM     | Alternative boosting model         |
| SHAP         | Explainable AI                     |
| Plotly       | Interactive visualization          |
| Streamlit    | Dashboard                          |
| Pytest       | Automated testing                  |
| Pylint       | Code quality                       |
| Space-Track  | Space-surveillance/CDM integration |
| ESA Kelvins  | Historical conjunction dataset     |

---

# 49. Project Structure

```text
Orbital-Sentinel-PBL/
│
├── data/
│
├── dashboard/
│
├── models_saved/
│
├── proofs/
│
├── scripts/
│
├── src/
│   └── orbital_sentinel/
│       ├── ingestion/
│       ├── preprocessing/
│       ├── features/
│       ├── models/
│       ├── physics/
│       ├── evaluation/
│       ├── fusion/
│       ├── uncertainty/
│       ├── explainability/
│       ├── events/
│       ├── monitoring/
│       └── validation/
│
├── tests/
│
├── .env.example
├── requirements.txt
└── README.md
```

---

# 50. Overall Technical Flow

```text
                 ESA KELVINS DATASET
                         ↓
                 DATA INGESTION
                         ↓
               SCHEMA / QUALITY CHECK
                         ↓
                  LEAKAGE CHECK
                         ↓
                 DATA CLEANING
                         ↓
                FEATURE ENGINEERING
                         ↓
             98 RAW + 46 ENGINEERED
                         ↓
                    144 FEATURES
                         ↓
                 EVENT-BASED SPLIT
                         ↓
                STANDARD SCALING
                         ↓
       ┌────────────────────────────────┐
       │        MACHINE LEARNING        │
       │                                │
       │ Ridge → Random Forest → XGBoost│
       └────────────────┬───────────────┘
                        ↓
                  RISK PREDICTION
                        ↓
                log10(Pc) prediction
                        ↓
          ┌─────────────┴─────────────┐
          ↓                           ↓
     PHYSICS ENGINE              UNCERTAINTY
          ↓                           ↓
      ANALYTIC Pc              Prediction Interval
          │                           │
          └─────────────┬─────────────┘
                        ↓
                   RISK FUSION
                        ↓
                   SHAP / XAI
                        ↓
               PLOTLY VISUALIZATION
                        ↓
               STREAMLIT DASHBOARD
                        ↓
               DECISION SUPPORT
```

---

# 51. Current Verified Results

```text
Training rows:             162,634
Testing rows:               24,484

Raw columns:                     103
Retained raw features:            98
Engineered features:              46
Final model features:             144

Training events:               10,524
Validation events:              2,630
Event overlap:                     0

Ridge validation R²:          0.5207
Random Forest validation R²:  0.8915
XGBoost validation R²:        ~0.9207

XGBoost features:                 144
Best iteration:                   499

High-risk threshold:            -5.0
Precision:                      ~0.698
Recall:                         ~0.423
F1:                             ~0.527
ROC-AUC:                        ~0.967

Physics validation sample:       20/20 successful
SHAP sample size:                  200

Automated tests:              81 passed
Pylint score:                  ~9.69/10
```

---

# 52. Project Significance

The key idea of Orbital Sentinel is not simply:

```text
Train ML → output risk
```

Instead, the system combines:

```text
Historical data
+
Machine learning
+
Orbital features
+
Uncertainty
+
Physics
+
Explainability
+
Interactive visualization
```

This creates a decision-support architecture for analysing space-object conjunction risk.

The ML component provides learned risk estimation.

The physics component provides physics-based analysis.

The uncertainty component communicates confidence and variability.

The explainability component provides insight into the model's reasoning.

The dashboard brings the complete analysis into one interface.

---

# 53. Future Development

Potential extensions include:

* Continuous live Space-Track monitoring
* Full historical CDM reconstruction
* High-fidelity orbital propagation
* 3D trajectory replay
* Advanced temporal modelling
* Physics-informed machine learning
* Multi-source surveillance-data fusion
* Multi-satellite constellation analysis
* Automated sensitivity analysis
* Probabilistic scenario simulation
* Dynamic uncertainty updates
* Human-in-the-loop operator feedback
* Large-scale catalogue monitoring
* More extensive ablation experiments
* Real-time deployment

---

# 54. Final System Concept

Orbital Sentinel is designed around the principle:

> **Use machine learning for rapid pattern-based risk estimation, physics for physical verification, uncertainty analysis for confidence, and explainable AI for transparency.**

The final system therefore treats collision-risk assessment as a combination of:

```text
DATA
  +
ML
  +
PHYSICS
  +
UNCERTAINTY
  +
EXPLAINABILITY
  +
VISUALIZATION
```

# Global Debries Count

|TYPE| Occupancy | Count |
|-----|----------|-------|
|Payloads | 59% | 18,697|
|Debris   |31% |  9,907|
|Rocket Bodies | 6% | 2,108|
|Unknown | 2% |661|


## Cleaned Data

<img width="440" height="285" alt="image" src="https://github.com/user-attachments/assets/31fe3a49-771b-4aa7-8d66-8e1ed0a78c70" />

## Columns in dataset
```python
  0 | event_id
  1 | time_to_tca
  2 | mission_id
  3 | risk
  4 | max_risk_estimate
  5 | max_risk_scaling
  6 | miss_distance
  7 | relative_speed
  8 | relative_position_r
  9 | relative_position_t
 10 | relative_position_n
 11 | relative_velocity_r
 12 | relative_velocity_t
 13 | relative_velocity_n
 14 | t_time_lastob_start
 15 | t_time_lastob_end
 16 | t_recommended_od_span
 17 | t_actual_od_span
 18 | t_obs_available
 19 | t_obs_used
 20 | t_residuals_accepted
 21 | t_weighted_rms
 22 | t_rcs_estimate
 23 | t_cd_area_over_mass
 24 | t_cr_area_over_mass
 25 | t_sedr
 26 | t_j2k_sma
 27 | t_j2k_ecc
 28 | t_j2k_inc
 29 | t_ct_r
 30 | t_cn_r
 31 | t_cn_t
 32 | t_crdot_r
 33 | t_crdot_t
 34 | t_crdot_n
 35 | t_ctdot_r
 36 | t_ctdot_t
 37 | t_ctdot_n
 38 | t_ctdot_rdot
 39 | t_cndot_r
 40 | t_cndot_t
 41 | t_cndot_n
 42 | t_cndot_rdot
 43 | t_cndot_tdot
 44 | c_object_type
 45 | c_time_lastob_start
 46 | c_time_lastob_end
 47 | c_recommended_od_span
 48 | c_actual_od_span
 49 | c_obs_available
 50 | c_obs_used
 51 | c_residuals_accepted
 52 | c_weighted_rms
 53 | c_rcs_estimate
 54 | c_cd_area_over_mass
 55 | c_cr_area_over_mass
 56 | c_sedr
 57 | c_j2k_sma
 58 | c_j2k_ecc
 59 | c_j2k_inc
 60 | c_ct_r
 61 | c_cn_r
 62 | c_cn_t
 63 | c_crdot_r
 64 | c_crdot_t
 65 | c_crdot_n
 66 | c_ctdot_r
 67 | c_ctdot_t
 68 | c_ctdot_n
 69 | c_ctdot_rdot
 70 | c_cndot_r
 71 | c_cndot_t
 72 | c_cndot_n
 73 | c_cndot_rdot
 74 | c_cndot_tdot
 75 | t_span
 76 | c_span
 77 | t_h_apo
 78 | t_h_per
 79 | c_h_apo
 80 | c_h_per
 81 | geocentric_latitude
 82 | azimuth
 83 | elevation
 84 | mahalanobis_distance
 85 | t_position_covariance_det
 86 | c_position_covariance_det
 87 | t_sigma_r
 88 | c_sigma_r
 89 | t_sigma_t
 90 | c_sigma_t
 91 | t_sigma_n
 92 | c_sigma_n
 93 | t_sigma_rdot
 94 | c_sigma_rdot
 95 | t_sigma_tdot
 96 | c_sigma_tdot
 97 | t_sigma_ndot
 98 | c_sigma_ndot
 99 | F10
100 | F3M
101 | SSN
102 | AP
```
