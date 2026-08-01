# Orbital-Sentinel-PBL
An Automated Machine Learning Triage System for Space Debris Conjunction Analysis

# Table of Contents
- [Abstract](#abstract)
- [Motivation](#motivation)
- [Space Situational Awareness Challenge](#space-situational-awareness-challenge)
- [Problem Statement](#problem-statement)
- [Research Gap](#research-gap)
- [Objectives](#objectives)
- [Key Contributions](#key-contributions)
- [System Overview](#system-overview)
- [Architecture Overview](#architecture-overview)
- [Design Principles](#design-principles)
- [Technology Stack](#technology-stack)
- [Data Sources](#data-sources)
- [Data Engineering Layer](#data-engineering-layer)
- [AI Prediction Layer](#ai-prediction-layer)
- [Physics Verification Layer](#physics-verification-layer)
- [Mission Intelligence Engine](#mission-intelligence-engine)
- [Digital Twin & Visualization](#digital-twin--visualization)
- [Explainable AI](#explainable-ai)
- [Decision Workflow](#decision-workflow)
- [Safety & Manual Verification Strategy](#safety--manual-verification-strategy)
- [Experimental Evaluation](#experimental-evaluation)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

**What is CDM??**
CDM stands for Conjunction Data Message. It is a standardized digital file format used to alert satellite operators when an active spacecraft and another orbiting object.

Sample CDM Message:
```python
CDM_ID                             =1605310377               
CREATED                            =2026-07-28 04:19:08.000000
EMERGENCY_REPORTABLE               =Y                        
TCA                                =2026-07-30T23:09:52.264000
MIN_RNG                            =3725                     
PC                                 =                         
SAT_1_ID                           =33471                    
SAT_1_NAME                         =SL-12 R/B(2)             
SAT1_OBJECT_TYPE                   =ROCKET BODY              
SAT1_RCS                           =LARGE                    
SAT_1_EXCL_VOL                     =3.00                     
SAT_2_ID                           =23048                    
SAT_2_NAME                         =SL-12 R/B(2)             
SAT2_OBJECT_TYPE                   =ROCKET BODY              
SAT2_RCS                           =LARGE                    
SAT_2_EXCL_VOL                     =3.00                     
```
### EMERGENCY_REPORTABLE = Y means that the predicted close approach (conjunction) between two space objects satisfies specific high-risk, emergency-reporting criteria (Y for Yes, N for No).

```
                           ┌──────────────────────────┐
                           │   Data Sources           │
                           │--------------------------│
                           │ ESA Kelvins Dataset      │
                           │ Live CCSDS CDMs          │
                           │ Space-Track (Future)     │
                           └─────────────┬────────────┘
                                         │
                                         ▼
                     ┌──────────────────────────┐
                     │ Data Ingestion Engine    │
                     │ • Load datasets          │
                     │ • Parse CDMs             │
                     │ • Validate               │
                     │ • Standardize            │
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │ Feature Engineering Engine │
                     │ • Relative Motion          │
                     │ • Orbital Features         │
                     │ • Covariance Features      │
                     │ • Object Features          │
                     │ • Space Weather            │
                     │ • Trend Features           │
                     └─────────────┬──────────────┘
                                   │
                                   ▼
                     ┌────────────────────────────┐
                     │ Data Preprocessing Engine  │
                     │ • Cleaning                 │
                     │ • Encoding                 │
                     │ • Scaling                  │
                     │ • Time Split               │
                     │ • SMOTE                    │
                     └─────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          AI Prediction Engine         Physics Engine
      (XGBoost / LightGBM)         (Monte Carlo Simulation)
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                     ┌────────────────────────────┐
                     │ Mission Intelligence Engine│
                     │ • Fusion                   │
                     │ • Confidence Analysis      │
                     │ • Recommendations          │
                     └─────────────┬──────────────┘
                                   ▼
                     ┌────────────────────────────┐
                     │ Explainability Engine      │
                     │ • SHAP                     │
                     │ • Feature Importance       │
                     │ • Decision Reasoning       │
                     └─────────────┬──────────────┘
                                   ▼
                     ┌──────────────────────────────┐
                     │ Dashboard / Digital Twin     │
                     │ • Alerts                     │
                     │ • 3D Orbit View              │
                     │ • Reports                    │
                     └──────────────────────────────┘
```

## Sources
https://kelvins.esa.int/collision-avoidance-challenge/data/
https://docs.poliastro.space/en/stable/
https://ui.adsabs.harvard.edu/abs/2021spde.confE.105D/abstract
https://geospatialworld.net/blogs/do-you-know-how-many-satellites-Earth/
https://www.discovermagazine.com/about-15-000-satellites-are-circling-earth-and-they-re-disrupting-the-sky-48550
https://time.com/article/2026/04/16/space-debris-satellites-growing-risk/



# Table of Contents

- Abstract
- Motivation
- Space Situational Awareness Challenge
- Problem Statement
- Research Gap
- Key Contributions
- System Overview
- Architecture Overview
- Design Principles
- Data Sources
- Data Engineering Layer
- AI Prediction Layer
- Physics Verification Layer
- Mission Intelligence Engine
- Digital Twin & Visualization
- Explainable AI
- Decision Workflow
- Safety & Manual Verification Strategy
- Experimental Evaluation
- Results
- Limitations
- Future Work
- References
