                  CONJUNCTION DATA
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
       ESA KELVINS              SPACE-TRACK
       Historical                Live/CDM
             │                       │
             └───────────┬───────────┘
                         ↓
                  DATA INGESTION
                         ↓
                 VALIDATION/CLEANING
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
       RF/Ridge      Models          Engine
          ↓              ↓              ↓
          └──────────────┼──────────────┘
                         ↓
                    RISK FUSION
                         ↓
                    UNCERTAINTY
                         ↓
                       SHAP
                         ↓
                DECISION SUPPORT
                         ↓
              STREAMLIT + PLOTLY
                         ↓
       ┌──────────┬──────────┬───────────┐
       ↓          ↓          ↓           ↓
     LIVE       FULL CDM    3D        WHAT-IF
     DATA       DETAILS     ORBIT      ANALYSIS