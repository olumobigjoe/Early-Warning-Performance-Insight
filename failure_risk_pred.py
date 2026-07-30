"""
GLT 302 Early Warning & Performance Insights App
--------------------------------------------------
A Streamlit app for secondary/tertiary school teachers to:
  1) Get quick data insights per academic year
  2) Predict which students are at risk of failing the final exam
  3) See what drives exam performance + actionable recommendations

Run with:  streamlit run app.py

Expected data schema (CSV/XLSX) for the "Predict in bulk" uploader:
  student_id, year, term, subject, practical_score, ca_score, exam_score
  (exam_score can be blank/NaN for students who haven't sat the exam yet)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ----------------------------------------------------------------------------
# PAGE CONFIG + THEME (cool blue / teal / violet palette)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Early Warning & Performance Insights",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#5B8DEF"      # cool blue
ACCENT = "#22D3C5"       # teal
VIOLET = "#8B7CF6"       # soft violet
DARK_BG = "#0F1B33"      # deep navy
CARD_BG = "#16233F"
LOW_COLOR = "#22D3C5"    # teal = low risk
MED_COLOR = "#F5B942"    # amber = medium risk
HIGH_COLOR = "#F2545B"   # coral red = high risk
TEXT_MUTED = "#A9B4CE"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, {DARK_BG} 0%, #a89c8e 100%);
        color: #EAF0FB;
    }}
    section[data-testid="stSidebar"] {{
        background: #9bb0cl;
    }}
    h1, h2, h3, h4 {{
        color: #EAF0FB !important;
        font-family: 'Trebuchet MS', sans-serif;
    }}
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid rgba(139,124,246,0.25);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }}
    .badge {{
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CARD_BG};
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        color: {TEXT_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {PRIMARY}, {VIOLET});
        color: white !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {ACCENT};
    }}
    .insight-box {{
        background: linear-gradient(135deg, rgba(91,141,239,0.15), rgba(139,124,246,0.12));
        border-left: 4px solid {PRIMARY};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

RISK_THRESHOLDS = {"Low": (0.0, 0.33), "Medium": (0.33, 0.66), "High": (0.66, 1.01)}
FAIL_CUTOFF = 10  # exam_score < 10 => "At Risk" label used for training


def risk_level_from_prob(p: float) -> str:
    if p < 0.33:
        return "Low"
    elif p < 0.66:
        return "Medium"
    return "High"


def risk_color(level: str) -> str:
    return {"Low": LOW_COLOR, "Medium": MED_COLOR, "High": HIGH_COLOR}[level]


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
REQUIRED_COLS = ["student_id", "name", "year", "term", "subject",
                  "practical_score", "ca_score", "exam_score"]


@st.cache_data
def load_default_data():
    df = pd.read_csv("glt302_scores.csv")
    # keep only the columns the app cares about, in the standard schema
    keep = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[keep].copy()
    for c in ["practical_score", "ca_score", "exam_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["student_id", "practical_score", "ca_score"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    for c in ["name", "year", "term", "subject", "exam_score"]:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["practical_score", "ca_score", "exam_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[REQUIRED_COLS]


# ----------------------------------------------------------------------------
# MODEL TRAINING (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def train_risk_model(df: pd.DataFrame):
    """Train an XGBoost early-warning model on rows with a known exam_score."""
    labeled = df.dropna(subset=["exam_score", "ca_score", "practical_score"]).copy()
    labeled["at_risk"] = (labeled["exam_score"] < FAIL_CUTOFF).astype(int)

    X = labeled[["practical_score", "ca_score"]].values
    y = labeled["at_risk"].values

    # class imbalance handling for XGBoost (ratio of negatives to positives)
    pos = max(y.sum(), 1)
    neg = max(len(y) - y.sum(), 1)
    scale_pos_weight = neg / pos

    # small, regularized tree ensemble - kept shallow to avoid overfitting on a
    # modest historical dataset (dozens to low hundreds of rows)
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    metrics = {}
    if len(labeled) >= 20 and labeled["at_risk"].nunique() == 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        metrics["accuracy"] = accuracy_score(y_test, preds)
        try:
            metrics["auc"] = roc_auc_score(y_test, probs)
        except ValueError:
            metrics["auc"] = np.nan
        # refit on full data for the deployed model
        model.fit(X, y)
    else:
        model.fit(X, y)
        metrics["accuracy"] = np.nan
        metrics["auc"] = np.nan

    # gain-based feature importance from XGBoost
    importance = pd.Series(
        model.feature_importances_, index=["practical_score", "ca_score"]
    )
    baseline_rate = labeled["at_risk"].mean()

    return {
        "model": model,
        "n_train": len(labeled),
        "metrics": metrics,
        "importance": importance,
        "baseline_rate": baseline_rate,
        "labeled": labeled,
    }


def predict_risk(bundle, practical_score, ca_score):
    X = np.array([[practical_score, ca_score]])
    prob = bundle["model"].predict_proba(X)[0, 1]
    return prob, risk_level_from_prob(prob)


# ----------------------------------------------------------------------------
# LOAD DATA + TRAIN MODEL
# ----------------------------------------------------------------------------
st.sidebar.title("🎓 Early Warning System")
st.sidebar.caption("GLT 302 · General Instrumentation")

data_source = st.sidebar.radio(
    "Data source",
    ["Use bundled historical data (2024-2026)", "Upload my own file"],
)

if data_source == "Upload my own file":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        help="Columns needed: student_id, year, term, subject, practical_score, ca_score, exam_score",
    )
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded) if uploaded.name.endswith("csv") else pd.read_excel(uploaded)
            df = validate_and_clean(raw)
            st.sidebar.success(f"Loaded {len(df)} rows.")
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
            df = load_default_data()
    else:
        st.sidebar.info("No file uploaded yet — showing bundled data.")
        df = load_default_data()
else:
    df = load_default_data()

bundle = train_risk_model(df)

st.title("🎓 Student Early Warning & Performance Dashboard")
st.caption("Predict exam risk early using CA & practical scores, and see what really drives performance.")

tab1, tab2, tab3 = st.tabs(["📊 Data Insights", "🔮 Risk Prediction", "💡 Recommendations"])

# ----------------------------------------------------------------------------
# TAB 1 — DATA INSIGHTS PER YEAR
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Quick insights, year by year")

    years = sorted(df["year"].dropna().unique())
    selected_year = st.selectbox("Select year", years, index=len(years) - 1 if years else 0)

    yr_df = df[df["year"] == selected_year]
    yr_scored = yr_df.dropna(subset=["exam_score"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students on record", len(yr_df))
    c2.metric("Students with exam scores", len(yr_scored))
    if len(yr_scored):
        c3.metric("Average exam score", f"{yr_scored['exam_score'].mean():.1f}")
        c4.metric("At-risk rate", f"{(yr_scored['exam_score'] < FAIL_CUTOFF).mean()*100:.1f}%")
    else:
        c3.metric("Average exam score", "—")
        c4.metric("At-risk rate", "—")

    if len(yr_scored) == 0:
        st.info(
            f"No exam scores recorded yet for {int(selected_year)} — this looks like the current, "
            "in-progress term. Once CA and practical scores are entered, use the **Risk Prediction** "
            "tab to flag at-risk students before the final exam."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                yr_scored, x="exam_score", nbins=15,
                title="Exam score distribution",
                color_discrete_sequence=[PRIMARY],
            )
            fig.add_vline(x=FAIL_CUTOFF, line_dash="dash", line_color=HIGH_COLOR,
                           annotation_text=f"Fail cutoff ({FAIL_CUTOFF})")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#EAF0FB")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(
                yr_scored, x="ca_score", y="exam_score", color="practical_score",
                title="CA score vs Exam score (colored by practical score)",
                color_continuous_scale=[ACCENT, PRIMARY, VIOLET],
            )
            fig2.add_hline(y=FAIL_CUTOFF, line_dash="dash", line_color=HIGH_COLOR)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#EAF0FB")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("##### Students most at risk this year")
        at_risk_tbl = (
            yr_scored[yr_scored["exam_score"] < FAIL_CUTOFF]
            .sort_values("exam_score")[["student_id", "name", "practical_score", "ca_score", "exam_score"]]
        )
        st.dataframe(at_risk_tbl, use_container_width=True, hide_index=True)

    st.markdown("##### Trend across all years")
    trend = (
        df.dropna(subset=["exam_score"])
        .groupby("year")
        .agg(avg_exam=("exam_score", "mean"),
             at_risk_pct=("exam_score", lambda x: (x < FAIL_CUTOFF).mean() * 100),
             n=("student_id", "count"))
        .reset_index()
    )
    if len(trend):
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=trend["year"], y=trend["at_risk_pct"],
                               name="At-risk rate (%)", marker_color=HIGH_COLOR, opacity=0.75))
        fig3.add_trace(go.Scatter(x=trend["year"], y=trend["avg_exam"],
                                   name="Average exam score", yaxis="y2",
                                   line=dict(color=ACCENT, width=3)))
        fig3.update_layout(
            yaxis=dict(title="At-risk rate (%)"),
            yaxis2=dict(title="Average exam score", overlaying="y", side="right"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#EAF0FB", legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2 — RISK PREDICTION
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Predict a student's exam risk")
    st.caption(
        f"Model trained on {bundle['n_train']} historical, fully-scored student records "
        f"(baseline at-risk rate: {bundle['baseline_rate']*100:.1f}%)."
    )
    if not np.isnan(bundle["metrics"].get("auc", np.nan)):
        m1, m2 = st.columns(2)
        m1.metric("Model accuracy (holdout)", f"{bundle['metrics']['accuracy']*100:.1f}%")
        m2.metric("Model AUC (holdout)", f"{bundle['metrics']['auc']:.2f}")

    pred_mode = st.radio("Predict for", ["A single student", "A batch of students (upload file)"], horizontal=True)

    if pred_mode == "A single student":
        name_input = st.text_input("Student name (optional)", "")
        c1, c2 = st.columns(2)
        practical = c1.slider("Practical score (out of 20)", 0.0, 20.0, 15.0, 0.5)
        ca = c2.slider("CA score (out of 20)", 0.0, 20.0, 13.0, 0.5)

        if st.button("Predict risk", type="primary"):
            prob, level = predict_risk(bundle, practical, ca)
            color = risk_color(level)
            label = name_input.strip() if name_input.strip() else "This student"
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center;">
                    <div style="font-size:1rem;color:{TEXT_MUTED};">{label} — probability of failing the exam</div>
                    <div style="font-size:2.6rem;font-weight:800;color:{color};">{prob*100:.1f}%</div>
                    <span class="badge" style="background:{color};color:#0F1B33;">{level} risk</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 33], "color": "rgba(34,211,197,0.25)"},
                        {"range": [33, 66], "color": "rgba(245,185,66,0.25)"},
                        {"range": [66, 100], "color": "rgba(242,84,91,0.25)"},
                    ],
                },
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#EAF0FB", height=280)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown(
            "Upload a file with at least `student_id`, `practical_score`, and `ca_score` columns."
        )
        batch_file = st.file_uploader("Upload CSV or Excel for batch prediction", type=["csv", "xlsx"], key="batch")
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file) if batch_file.name.endswith("csv") else pd.read_excel(batch_file)
            if not {"student_id", "practical_score", "ca_score"}.issubset(batch_df.columns):
                st.error("File must include: student_id, practical_score, ca_score")
            else:
                probs, levels = [], []
                for _, row in batch_df.iterrows():
                    p, lv = predict_risk(bundle, row["practical_score"], row["ca_score"])
                    probs.append(p)
                    levels.append(lv)
                batch_df["fail_probability"] = np.round(probs, 3)
                batch_df["risk_level"] = levels

                cols_order = [c for c in ["student_id", "name", "practical_score", "ca_score",
                                           "fail_probability", "risk_level"] if c in batch_df.columns]
                other_cols = [c for c in batch_df.columns if c not in cols_order]
                batch_df = batch_df[cols_order + other_cols]

                def highlight_risk(val):
                    color_map = {"Low": LOW_COLOR, "Medium": MED_COLOR, "High": HIGH_COLOR}
                    return f"background-color: {color_map.get(val, '')}; color: #0F1B33; font-weight:700;"

                st.dataframe(
                    batch_df.style.applymap(highlight_risk, subset=["risk_level"]),
                    use_container_width=True, hide_index=True,
                )
                st.download_button(
                    "Download predictions as CSV",
                    batch_df.to_csv(index=False).encode("utf-8"),
                    "risk_predictions.csv",
                    "text/csv",
                )

# ----------------------------------------------------------------------------
# TAB 3 — RECOMMENDATIONS
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("What drives exam performance?")

    imp = bundle["importance"].sort_values()
    fig4 = px.bar(
        imp, orientation="h",
        title="Feature importance (XGBoost gain-based importance)",
        labels={"value": "Relative importance", "index": "Feature"},
        color=imp.values,
        color_continuous_scale=[ACCENT, PRIMARY, VIOLET],
    )
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#EAF0FB", showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    labeled = bundle["labeled"]
    corr = labeled[["practical_score", "ca_score", "exam_score"]].corr()["exam_score"].drop("exam_score")
    grp = labeled.groupby("at_risk")[["practical_score", "ca_score"]].mean()

    st.markdown("##### Correlation with final exam score")
    cc1, cc2 = st.columns(2)
    cc1.metric("Practical score", f"r = {corr.get('practical_score', float('nan')):.2f}")
    cc2.metric("CA score", f"r = {corr.get('ca_score', float('nan')):.2f}")

    st.markdown("### 📌 Actionable insights for teachers")

    practical_gap = grp.loc[0, "practical_score"] - grp.loc[1, "practical_score"] if 1 in grp.index and 0 in grp.index else np.nan
    ca_gap = grp.loc[0, "ca_score"] - grp.loc[1, "ca_score"] if 1 in grp.index and 0 in grp.index else np.nan

    st.markdown(
        f"""
        <div class="insight-box">
        <b>1. Practical performance is the single strongest predictor of exam success.</b><br>
        Practical score correlates most strongly with exam score (r ≈ {corr.get('practical_score', 0):.2f}).
        Students who go on to fail score on average {practical_gap:.1f} points lower on practicals
        than those who pass. Prioritize hands-on lab remediation and one-on-one demonstration time
        for any student scoring below the practical median.
        </div>
        <div class="insight-box">
        <b>2. CA score is the earliest and most actionable warning signal.</b><br>
        Students who eventually fail score on average {ca_gap:.1f} points lower in continuous
        assessment (r ≈ {corr.get('ca_score', 0):.2f}) — and CA scores are usually available weeks
        before the final exam. Run this model right after CA is recorded, not after the exam,
        so intervention still has time to work.
        </div>
        <div class="insight-box">
        <b>3. Historically, about {bundle['baseline_rate']*100:.0f}% of students in this course fail
        the final exam</b> — a rate high enough to justify a standing mid-semester checkpoint.
        Flag every "Medium" or "High" risk student from the Prediction tab for a structured
        catch-up session (extra practicals + a CA resit/tutorial) before the exam, rather than
        waiting for results day.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Currently flagged at-risk students (from bundled/uploaded data)")
    unscored = df[df["exam_score"].isna() & df["ca_score"].notna() & df["practical_score"].notna()]
    if len(unscored):
        probs, levels = [], []
        for _, row in unscored.iterrows():
            p, lv = predict_risk(bundle, row["practical_score"], row["ca_score"])
            probs.append(p)
            levels.append(lv)
        unscored = unscored.copy()
        unscored["fail_probability"] = np.round(probs, 3)
        unscored["risk_level"] = levels
        st.dataframe(
            unscored.sort_values("fail_probability", ascending=False)[
                ["student_id", "name", "year", "practical_score", "ca_score", "fail_probability", "risk_level"]
            ],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info(
            "No students currently have CA/practical scores without an exam score yet — "
            "once the current term's CA and practical scores are entered, they'll show up here automatically."
        )

st.sidebar.markdown("---")
st.sidebar.caption("Built for early intervention — flag risk while there's still time to act.")
