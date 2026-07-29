# 🎓 Student Early Warning & Performance Insights

A Streamlit app that flags students at risk of failing their final exam **before** the exam happens — using only the scores teachers already have (Continuous Assessment + Practicals) — and explains what actually drives exam performance so teachers know where to intervene.

Built and validated on 3 years of real score sheets (2024–2026) from a Nigerian polytechnic course (GLT 302 – General Instrumentation).

![Python](https://img.shields.io/badge/Python-3.10+-5B8DEF?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-22D3C5?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-8B7CF6?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚨 The Problem

By the time a final exam result comes back, it's too late to help the student who failed it. Teachers already collect CA and practical scores weeks before the exam — but rarely use them as an early-warning signal. This project turns that existing data into a live risk score, automatically.

## ✅ What It Does

| | |
|---|---|
| **📊 Data Insights** | Per-year breakdown — average scores, at-risk rate, score distributions, CA-vs-exam scatter, and a multi-year trend view |
| **🔮 Risk Prediction** | Enter a student's CA + practical score (or upload a whole class) to get a fail-probability and a Low / Medium / High risk badge, powered by XGBoost |
| **💡 Recommendations** | Feature importance ranking + 3 data-driven, actionable insights for teachers, plus an auto-flagged list of current students who have CA/practical scores but no exam score yet |

## 📈 What the Data Actually Shows

Trained on **267 real, fully-scored historical records** across three cohorts (2024, 2025, 2026):

| Year | Students | Fully scored | Avg. exam score | At-risk rate |
|------|---------:|--------------:|------------------:|--------------:|
| 2024 | 70 | 70 | 46.5 | 37.1% |
| 2025 | 48 | 48 | 43.4 | 41.7% |
| 2026 | 151 | 149 | 40.3 | 47.7% |

- **Practical score** correlates most strongly with exam outcome (r ≈ 0.57)
- **CA score** also matters (r ≈ 0.36), and is available earliest
- **Overall historical at-risk rate: 44.2%** — trending upward year over year

## 🧠 Model

- **Algorithm:** XGBoost classifier (`XGBClassifier`), shallow & regularized (`max_depth=3`, `n_estimators=150`, `reg_lambda=2.0`) — validated at **~76% holdout accuracy, 0.78 AUC**
- **Features:** `practical_score`, `ca_score`
- **Target:** `exam_score < 40` → "At Risk"
- **Class imbalance:** handled via `scale_pos_weight`
- **Output:** calibrated fail-probability → mapped to `Low` (<33%), `Medium` (33–66%), `High` (>66%) risk bands

## 🗂️ Data Schema

The app expects (and the bundled `glt302_scores.csv` provides) this schema:

```
student_id, name, year, term, subject, practical_score, ca_score, exam_score
```

`exam_score` can be blank for students who haven't sat the exam yet — those rows automatically show up in the **Prediction** and **Recommendations** tabs as "needs a risk check."

## 🚀 Getting Started

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

Keep `glt302_scores.csv` in the same folder as `app.py`, or use the in-app uploader to swap in your own class data (CSV or Excel) — no code changes needed.

## 📁 Project Structure

```
.
├── app.py                # Streamlit app (data insights, prediction, recommendations)
├── glt302_scores.csv     # Cleaned, bundled historical dataset (2024-2026)
├── requirements.txt      # Python dependencies
└── README.md
```

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [XGBoost](https://xgboost.readthedocs.io/) — risk prediction model
- [scikit-learn](https://scikit-learn.org/) — train/test split & evaluation metrics
- [Plotly](https://plotly.com/python/) — interactive charts
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling

## 🔮 Roadmap Ideas

- [ ] Add more predictive features as they become available (attendance, assignment scores)
- [ ] Multi-subject / multi-course support with per-subject models
- [ ] Automated email/SMS alerts to teachers for High-risk students
- [ ] Model explainability (SHAP values) per student, not just global importance

## 📄 License

MIT — free to use and adapt for your school or institution.

---

*Built for early intervention — flag risk while there's still time to act.*
