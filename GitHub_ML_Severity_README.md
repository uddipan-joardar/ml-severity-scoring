# 📊 ML Severity Scoring & Multilingual Quality Intelligence

> **Two-stage stacking ensemble ML system processing 190K+ multilingual quality notifications**  
> XGBoost · LightGBM · Random Forest · SVM · Azure OpenAI · Python · MLflow

---

## 🎯 Business Problem

A global manufacturing client receives **190,000+ quality notifications per year** in multiple languages. Manual triage to identify critical vs. routine issues was slow, inconsistent, and consumed significant analyst time — with high-severity issues sometimes delayed due to volume.

## ✅ Solution

An end-to-end **ML + GenAI intelligence pipeline** that:
1. Translates multilingual notifications to English via Azure OpenAI with autonomous quality checks
2. Classifies each notification as critical vs. non-critical (binary stage)
3. Assigns specific severity level (multiclass stage)
4. Routes high-severity cases for immediate escalation automatically

## 📊 Impact

| Metric | Result |
|---|---|
| Manual review workload | **60% reduction** |
| Records processed | **190,000+ notifications** |
| Languages handled | Multilingual (auto-translated) |
| Escalation | Automated for high-severity cases |

---

## 🏗️ ML Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              RAW MULTILINGUAL NOTIFICATIONS                   │
│                   (190K+ records/year)                        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│         AZURE OPENAI TRANSLATION PIPELINE                     │
│  • Translate to English                                       │
│  • Confidence scoring on output                               │
│  • Autonomous retry on low-confidence translations            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                              │
│  • Anomaly detection scores                                   │
│  • Temporal encoding (day, month, quarter patterns)           │
│  • Historical aggregations per product/line/supplier          │
│  • Text features (TF-IDF, notification category encoding)     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│         STAGE 1: BINARY CLASSIFICATION                        │
│    Stacking Ensemble (Random Forest + XGBoost + LightGBM     │
│                     + SVM + KNN + Ridge)                      │
│    → Critical vs Non-Critical                                 │
└──────────────────────┬───────────────────────────────────────┘
                       │ Critical cases only
                       ▼
┌──────────────────────────────────────────────────────────────┐
│         STAGE 2: MULTICLASS SEVERITY PREDICTION               │
│    Stacking Ensemble (same 6 models, retrained for           │
│    multiclass: Severity 1 / 2 / 3 / 4)                       │
│    Cost-sensitive learning to maximise recall on S1          │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              ROUTING & ESCALATION                             │
│  S1/S2 → Immediate escalation alert                          │
│  S3/S4 → Standard review queue                               │
│  Non-critical → Auto-closed with log                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost, LightGBM, Random Forest, SVM, KNN, Ridge |
| Ensemble | Scikit-learn StackingClassifier |
| Translation | Azure OpenAI GPT-4o |
| Feature Engineering | Pandas, NumPy, Scikit-learn |
| Imbalance Handling | Cost-sensitive learning, class weights |
| Experiment Tracking | MLflow |
| Language | Python 3.10+ |

---

## 📁 Repository Structure

```
ml-severity-scoring/
├── data/
│   └── sample_data.csv            # Anonymised sample (real data not included)
├── translation/
│   └── azure_translator.py        # OpenAI translation with retry logic
├── features/
│   ├── feature_engineering.py     # Anomaly scores, temporal, historical features
│   └── text_features.py           # TF-IDF and categorical encoding
├── models/
│   ├── binary_classifier.py       # Stage 1: critical vs non-critical
│   ├── severity_classifier.py     # Stage 2: multiclass severity
│   └── stacking_ensemble.py       # Ensemble architecture
├── evaluation/
│   ├── metrics.py                 # Precision, recall, F1, AUC
│   └── threshold_analysis.py      # Business risk threshold calibration
├── mlflow_tracking/
│   └── experiment_runner.py       # MLflow experiment logging
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## 🔑 Key Engineering Decisions

**Why two-stage instead of single multiclass?**
A single multiclass model struggles when the binary signal (critical vs. not) is the strongest separator. Two-stage allows the first model to specialise in rejection of non-critical noise before the severity model focuses only on meaningful cases — improving recall on high-severity categories.

**Why cost-sensitive learning over SMOTE?**
SMOTE synthesises data in feature space, which can introduce unrealistic samples for tabular operational data. Cost-sensitive weights directly penalise misclassification of critical cases during training without altering the data distribution.

**Why agent-like retry on translation?**
Translation confidence varies significantly across languages and notification formats. Autonomous retry on low-confidence outputs ensures ML scoring always uses clean, verified English text — preventing translation errors from cascading into wrong severity predictions.

---

## 📬 Contact

**Uddipan Joardar** — [LinkedIn](https://linkedin.com/in/uddipan-joardar38302580) · [Email](mailto:dip.joardar@gmail.com)
