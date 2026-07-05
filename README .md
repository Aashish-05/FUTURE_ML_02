# 🎫 Smart Ticket Triage System

Automatically classify customer support tickets by **category** and **priority** using NLP and Machine Learning — with a rigorous, honest evaluation of what the data can actually support.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-enabled-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-informational)

---

## 📌 Project Overview

This project builds an end-to-end ticket-triage pipeline from raw text to a working classifier, covering:

- 🏷️ **Category classification** — Billing inquiry, Technical issue, Refund request, Cancellation request, Product inquiry
- ⚡ **Priority prediction** — Critical / High / Medium / Low
- 🔍 **Pre-modeling data audit** — chi-square tests to check whether the labels are statistically learnable *before* training anything
- 🧹 **Dependency-free text cleaning** — regex + scikit-learn stop words, no external corpus downloads
- 🤖 **4 model families benchmarked** — Logistic Regression, Linear SVM, Random Forest, XGBoost — with `GridSearchCV` tuning
- 📊 **Executive dashboard** — KPI cards, confusion matrices, cross-validation charts, word clouds
- 💾 **Exported, reusable artifacts** — trained models, vectorizers, encoders (pickle)

Built for: Future Interns — Machine Learning Task 2 (2026)

---

## 📊 Results

| Task | Best model | Accuracy | F1 (weighted) | Chance baseline |
|------|-----------|----------|----------------|------------------|
| Category (5 classes) | Linear SVM | 21.5% | 0.2148 | ~20% |
| Priority (4 classes) | Linear SVM | 26.0% | 0.2597 | ~25% |

**Model comparison — Category classification**

| Model | Accuracy | F1 (weighted) |
|-------|----------|----------------|
| ⭐ Linear SVM | 0.2149 | 0.2148 |
| Logistic Regression | 0.2090 | 0.2083 |
| Naive Bayes | 0.2019 | 0.2018 |
| XGBoost | 0.2019 | 0.1995 |

**Model comparison — Priority classification**

| Model | Accuracy | F1 (weighted) |
|-------|----------|----------------|
| ⭐ Linear SVM | 0.2597 | 0.2597 |
| Tuned Logistic Regression | 0.2503 | 0.2502 |
| Logistic Regression | 0.2485 | 0.2486 |
| XGBoost | 0.2456 | 0.2449 |
| Random Forest | 0.2444 | 0.2439 |

Both tasks land close to their random-guess baseline. A chi-square audit run *before* training (see notebook, Section 2) shows none of the leakage-free fields are statistically associated with either label (all p > 0.5) — meaning this is a property of the dataset's labels, not a modeling shortfall. Four model families and a tuned grid search all converge on the same ceiling, which is itself evidence for that conclusion. Full reasoning and next steps are in the notebook.

---

## 🗂️ Repository Structure

```
smart-ticket-triage/
│
├── 📓 Smart_Ticket_Triage.ipynb     ← Full notebook, executed with outputs
├── 📄 customer_support_tickets.csv  ← Dataset (8,469 tickets)
├── 📄 model_results.csv             ← Accuracy / F1 for every model tried
│
├── 📊 fig_overview.png              ← Ticket volume EDA (type, priority, channel, age)
├── 📊 fig_wordclouds.png            ← Vocabulary by ticket category
├── 📊 fig_confusion.png             ← Category + priority confusion matrices
├── 📊 fig_cv.png                    ← 5-fold cross-validation stability
├── 📊 fig_terms.png                 ← Top TF-IDF terms per category
├── 📊 fig_dashboard.png             ← Full KPI scorecard dashboard
│
├── 📁 artifacts/                    ← Saved models, vectorizers, label encoders (.pkl)
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🗃️ Dataset

[Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) (Kaggle) — 8,469 rows, 17 columns.

| Field group | Columns |
|---|---|
| Text | `Ticket Description`, `Ticket Subject` |
| Metadata (safe — used) | `Ticket Type`, `Ticket Channel`, `Product Purchased`, `Customer Age`, `Customer Gender`, `Date of Purchase` |
| Post-resolution (excluded — leakage) | `Ticket Status`, `Resolution`, `First Response Time`, `Time to Resolution`, `Customer Satisfaction Rating` |

**Category classes (5):** Billing inquiry · Technical issue · Refund request · Cancellation request · Product inquiry
**Priority classes (4):** Critical · High · Medium · Low

---

## 🔧 Tech Stack

| Layer | Library / Tool |
|---|---|
| Language | Python 3.10 |
| Text cleaning | `re` + `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` (no NLTK downloads) |
| Features | `TfidfVectorizer` (word 1–2 grams) + engineered urgency/recency features |
| Models | `LogisticRegression`, `LinearSVC`, `RandomForestClassifier`, `XGBClassifier` |
| Tuning / validation | `GridSearchCV`, `StratifiedKFold`, `cross_val_score` |
| Imbalance handling | `class_weight='balanced'`, `imblearn.RandomOverSampler` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| Stats | `scipy.stats.chi2_contingency` |
| Export | `pickle`, `zipfile` |

---

## 🧹 Feature Engineering Pipeline

```
Raw Ticket Description
      │
      ▼   normalize_ticket_text()  — strip {placeholders}, URLs, emails, numbers
      │                               lowercase, remove stop words, filter short tokens
      ▼   urgency_score()          — counts urgent/broken/refund/angry-type terms
      │
      ▼   TfidfVectorizer()        — 1–2 gram term weights
      │
      ▼   + one-hot metadata + scaled numeric features (priority model only)
      │
      ▼   ML Model → category / priority prediction + confidence
```

Example from the notebook:

```
Input     : "My laptop keyboard stopped working after the latest update, this is urgent!"
Normalized: "laptop keyboard stopped working latest update urgent"
Urgency terms detected: 1

→ Category : Technical issue
→ Priority : MEDIUM
```

---

## 📓 Notebook Walkthrough

| Section | What it does |
|---|---|
| 1. Load the Data | Reads the CSV, shows shape and a sample of tickets |
| 2. Data Audit | Chi-square tests — checks whether labels are statistically learnable *before* modeling |
| 3. Feature Engineering | Text normalization, urgency scoring, recency features, word clouds |
| 4. Category Model | TF-IDF + 4 models compared, classification report |
| 5. Priority Model | TF-IDF + metadata features, imbalance handling, `GridSearchCV` tuning |
| 6. Diagnostics | Confusion matrices, 5-fold cross-validation stability |
| 7. Interpretability | Top TF-IDF terms per category (logistic regression coefficients) |
| 8. Executive Dashboard | KPI cards + all charts in one dashboard figure |
| — What This Tells Us | Honest interpretation of results, tied back to the Section 2 audit |
| 9. Live Inference | `classify_ticket()` — end-to-end prediction on new ticket text |
| 10. Export | Saves models/vectorizers/encoders, zips all outputs |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/smart-ticket-triage.git
cd smart-ticket-triage

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Smart_Ticket_Triage.ipynb
```

No external downloads (e.g. NLTK corpora) are needed — everything runs offline.

---

## 🤖 Live Inference

```python
result = classify_ticket(
    "My account was charged twice. I need an immediate refund.",
    subject="Payment issue", channel="Chat", product="Apple AirPods"
)
# Returns:
# {
#   'category': 'Refund request',
#   'category_confidence': '24%',
#   'priority': 'Medium',
#   'priority_badge': 'MEDIUM',
#   'urgency_score': 1
# }
```

---

## 💾 Exported Artifacts

```
artifacts/
├── vectorizer_category.pkl     ← TF-IDF vectorizer (category model)
├── vectorizer_priority.pkl     ← TF-IDF vectorizer (priority model)
├── onehot_priority.pkl         ← One-hot encoder for metadata
├── scaler_priority.pkl         ← StandardScaler for numeric features
├── model_category.pkl          ← Best category classifier (Linear SVM)
├── model_priority.pkl          ← Best priority classifier (Linear SVM)
├── label_encoder_category.pkl  ← LabelEncoder for category classes
└── label_encoder_priority.pkl  ← LabelEncoder for priority classes
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
wordcloud
scipy
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📄 License

Open-source under the MIT License.
