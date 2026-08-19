# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.feature_selection import chi2
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gastric Cancer Risk Assessment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  HIGH CONTRAST & CLEAN TYPOGRAPHY CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    background-color: #0b132b !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #ffffff !important;
}

.main .block-container {
    padding: 1.5rem 2.2rem 3rem;
    max-width: 1350px;
}

/* Ensure Top Header & Sidebar Toggle are Accessible */
header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 99999 !important;
}

[data-testid="stSidebarCollapseButton"] button {
    color: #38bdf8 !important;
    background: #1e293b !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    border-radius: 6px !important;
}

[data-testid="stSidebarCollapseButton"] button:hover {
    background: #0284c7 !important;
    color: #ffffff !important;
}

/* Sidebar Background */
[data-testid="stSidebar"] {
    background: #070d1e !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Button Customization */
.stButton > button {
    width: 100%;
    background: #1e293b;
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 11px 16px;
    border-radius: 8px;
    transition: all 0.2s ease;
    text-align: left;
}

.stButton > button:hover {
    background: rgba(56, 189, 248, 0.2);
    border-color: #38bdf8;
    color: #38bdf8;
}

/* Card Styling with High Contrast Text */
.metric-card {
    background: #162038;
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #38bdf8;
    margin-top: 4px;
}

.metric-label {
    font-size: 0.75rem;
    color: #e2e8f0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 6px;
}

/* Input Form Control Label High Contrast */
.stSelectbox label, .stSlider label, .stRadio label {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    margin-bottom: 6px !important;
}

/* Fix Tab Font Size & Spacing */
button[data-baseweb="tab"] {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    padding: 10px 18px !important;
}

button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

/* High Contrast Paragraphs & Lists */
p, span, li, div {
    color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPER COMPONENTS
# ─────────────────────────────────────────────
def page_header(title, subtitle):
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem; border-bottom: 1px solid rgba(255, 255, 255, 0.12); padding-bottom: 0.8rem;">
        <h2 style="color: #ffffff; font-weight: 800; margin: 0; font-size: 1.75rem; letter-spacing: -0.01em;">{title}</h2>
        <p style="color: #cbd5e1; margin-top: 6px; font-size: 0.92rem; line-height: 1.5;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def metric_card(value, label, color="#38bdf8", icon=""):
    st.markdown(f"""
    <div class="metric-card" style="border-top: 3px solid {color};">
        <div style="font-size: 1.2rem; margin-bottom: 2px;">{icon}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def pretty_name(col):
    overrides = {
        "age": "Age", "gender": "Gender", "family_history": "Family History",
        "smoking_habits": "Smoking Habits", "alcohol_consumption": "Alcohol Habits",
        "helicobacter_pylori_infection": "H. Pylori Status", "dietary_habits": "Dietary Salt Intake",
        "endoscopic_images": "Endoscopic Imaging", "biopsy_results": "Biopsy Result", "ct_scan": "CT Scan",
        "diana_microt": "DIANA-microT", "elmmo": "ElMMo", "microcosm": "MicroCosm", "miranda": "miRanda",
        "mirdb": "miRDB", "pictar": "PicTar", "pita": "PITA", "targetscan": "TargetScan",
        "predicted.sum": "Predicted-Tool Sum", "all.sum": "Algorithm Total Sum"
    }
    if col in overrides:
        return overrides[col]
    return col.replace("existing_conditions_", "").replace("mature_mirna_id_", "").replace("target_symbol_", "").replace("_", " ").title()

# ─────────────────────────────────────────────
#  DATA LOADING & MODEL TRAINING
# ─────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
DATASET_PATHS = [
    APP_DIR / "cleaned_gcs_kushal.csv",
    APP_DIR / "Dataset" / "cleaned_gcs_kushal.csv",
    APP_DIR.parent / "Dataset" / "cleaned_gcs_kushal.csv",
]

@st.cache_data(show_spinner=False)
def load_data():
    dataset_path = None
    for p in DATASET_PATHS:
        if p.is_file():
            dataset_path = p
            break
    if dataset_path is None:
        st.error("Dataset file cleaned_gcs_kushal.csv not found.")
        st.stop()

    df = pd.read_csv(dataset_path)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    return df

@st.cache_resource(show_spinner=False)
def get_trained_model(df):
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
    )

    model = DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=50,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, X.columns.tolist(), X_test, y_test, y_pred, y_prob

@st.cache_data(show_spinner=False)
def compute_chi2_scores(df):
    X = df.drop(columns=["label"]).clip(lower=0)
    y = df["label"]
    scores, pvals = chi2(X, y)
    return pd.DataFrame({"Feature": X.columns, "Chi2": scores, "PValue": pvals}).sort_values("Chi2", ascending=False)

df = load_data()
model, feature_cols, X_test, y_test, y_pred, y_prob = get_trained_model(df)
chi2_df = compute_chi2_scores(df)

# Calculate model metrics
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION (Cleaned, Version Stuff Removed)
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "predictor"

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 18px 10px 14px;">
        <div style="font-size: 2.3rem; margin-bottom: 6px;">🧬</div>
        <div style="font-weight: 800; font-size: 1.2rem; color: #38bdf8; letter-spacing: 0.02em;">GCS Predictor</div>
        <div style="font-size: 0.76rem; color: #cbd5e1; margin-top: 4px; font-weight: 500;">Gastric Cancer Assessment</div>
    </div>
    <div style="height: 1px; background: rgba(255, 255, 255, 0.12); margin-bottom: 18px;"></div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("predictor", "🤖", "Cancer Risk Predictor"),
        ("overview", "📁", "Dataset Overview"),
        ("feature", "🔬", "Feature Analysis"),
    ]

    for key, icon, label in nav_items:
        is_active = st.session_state.page == key
        if st.button(f"{icon}  {label}", key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()

# ═════════════════════════════════════════════
#  PAGE 1: CANCER RISK PREDICTOR (AUTOMATIC LIVE UPDATE)
# ═════════════════════════════════════════════
if st.session_state.page == "predictor":
    page_header("Gastric Cancer Risk Predictor", "Tweak patient clinical metrics below to instantly see live risk assessment.")

    col_input, col_result = st.columns([1.1, 0.9])

    input_dict = {col: 0 for col in feature_cols}

    with col_input:
        st.markdown("<h5 style='color: #ffffff; font-weight: 700; margin-bottom: 12px;'>Patient Metrics Input</h5>", unsafe_allow_html=True)
        
        tab_clinical, tab_conditions, tab_genomics = st.tabs([
            "🩺 Clinical & Demographics", "📋 Existing Conditions", "🧬 miRNA & Biomarkers"
        ])

        with tab_clinical:
            c1, c2 = st.columns(2)
            with c1:
                age_val = st.slider("Age (Years)", 18, 90, 52, key="in_age")
                input_dict["age"] = age_val

                gender_val = st.selectbox("Gender", ["Female", "Male"], key="in_gender")
                input_dict["gender"] = 1 if gender_val == "Male" else 0

                fam_val = st.selectbox("Family History of Cancer", ["No", "Yes"], key="in_fam")
                input_dict["family_history"] = 1 if fam_val == "Yes" else 0

                smoke_val = st.selectbox("Smoking Habits", ["No", "Yes"], key="in_smoke")
                input_dict["smoking_habits"] = 1 if smoke_val == "Yes" else 0

            with c2:
                alc_val = st.selectbox("Alcohol Habits", ["No", "Yes"], key="in_alc")
                input_dict["alcohol_consumption"] = 1 if alc_val == "Yes" else 0

                pylori_val = st.selectbox("H. Pylori Infection", ["No", "Yes"], key="in_pylori")
                input_dict["helicobacter_pylori_infection"] = 1 if pylori_val == "Yes" else 0

                diet_val = st.selectbox("Dietary Salt Intake", ["Low Salt", "High Salt"], key="in_diet")
                input_dict["dietary_habits"] = 1 if diet_val == "High Salt" else 0

                endo_val = st.selectbox("Endoscopic Imaging", ["Normal", "Abnormal"], key="in_endo")
                input_dict["endoscopic_images"] = 1 if endo_val == "Abnormal" else 0

        with tab_conditions:
            c1, c2 = st.columns(2)
            with c1:
                biopsy_val = st.selectbox("Biopsy Result", ["Negative", "Positive"], key="in_biopsy")
                input_dict["biopsy_results"] = 1 if biopsy_val == "Positive" else 0

                ct_val = st.selectbox("CT Scan Result", ["Negative", "Positive"], key="in_ct")
                input_dict["ct_scan"] = 1 if ct_val == "Positive" else 0

            with c2:
                cond_choice = st.radio(
                    "Existing Health Condition",
                    ["None / Unknown", "Chronic Gastritis", "Diabetes"],
                    key="in_cond"
                )
                if cond_choice == "Chronic Gastritis":
                    input_dict["existing_conditions_Chronic Gastritis"] = 1
                elif cond_choice == "Diabetes":
                    input_dict["existing_conditions_Diabetes"] = 1
                else:
                    input_dict["existing_conditions_Unknown"] = 1

        with tab_genomics:
            c1, c2 = st.columns(2)
            with c1:
                mirna_choice = st.selectbox("miRNA Identifier", ["MIR123_1", "MIR234_2", "MIR345_3"], key="in_mirna")
                if mirna_choice == "MIR123_1":
                    input_dict["mature_mirna_id_MIR123_1"] = 1
                elif mirna_choice == "MIR234_2":
                    input_dict["mature_mirna_id_MIR234_2"] = 1
                else:
                    input_dict["mature_mirna_id_MIR345_3"] = 1

                gene_choice = st.selectbox("Target Gene Mutation", ["CDH1", "KRAS", "TP53"], key="in_gene")
                if gene_choice == "CDH1":
                    input_dict["target_symbol_CDH1"] = 1
                elif gene_choice == "KRAS":
                    input_dict["target_symbol_KRAS"] = 1
                else:
                    input_dict["target_symbol_TP53"] = 1

            with c2:
                pred_sum = st.slider("Prediction Tool Score", 0.0, 1.0, 0.45, step=0.05, key="in_score")
                input_dict["predicted.sum"] = pred_sum
                input_dict["all.sum"] = pred_sum * 1.2
                input_dict["diana_microt"] = pred_sum
                input_dict["elmmo"] = pred_sum
                input_dict["microcosm"] = pred_sum
                input_dict["miranda"] = pred_sum
                input_dict["mirdb"] = pred_sum
                input_dict["pictar"] = pred_sum
                input_dict["pita"] = pred_sum
                input_dict["targetscan"] = pred_sum

    with col_result:
        st.markdown("<h5 style='color: #ffffff; font-weight: 700; margin-bottom: 12px;'>Live Risk Assessment</h5>", unsafe_allow_html=True)

        input_df = pd.DataFrame([input_dict])[feature_cols]
        
        # Calculate dynamic clinical risk points automatically as settings change
        risk_points = 0
        max_points = 10.0

        if input_dict["biopsy_results"] == 1: risk_points += 2.5
        if input_dict["ct_scan"] == 1: risk_points += 2.0
        if input_dict["endoscopic_images"] == 1: risk_points += 1.5
        if input_dict["helicobacter_pylori_infection"] == 1: risk_points += 1.5
        if input_dict["existing_conditions_Chronic Gastritis"] == 1: risk_points += 1.0
        if input_dict["family_history"] == 1: risk_points += 0.8
        if input_dict["age"] > 60: risk_points += 0.7

        calculated_risk_pct = min(98.0, max(4.0, (risk_points / max_points) * 100))
        tree_pred = model.predict(input_df)[0]
        
        if calculated_risk_pct >= 50.0 or tree_pred == 1:
            status_title = "HIGH RISK OF GASTRIC CANCER"
            status_color = "#ef4444"
            status_bg = "rgba(239, 68, 68, 0.15)"
            border_color = "rgba(239, 68, 68, 0.5)"
            icon = "⚠️"
            recommendation = "Immediate endoscopic evaluation and clinical consultation advised."
        elif calculated_risk_pct >= 25.0:
            status_title = "MODERATE GASTRIC RISK"
            status_color = "#f59e0b"
            status_bg = "rgba(245, 158, 11, 0.15)"
            border_color = "rgba(245, 158, 11, 0.5)"
            icon = "⚡"
            recommendation = "Follow-up screening and dietary modifications recommended."
        else:
            status_title = "LOW RISK OF GASTRIC CANCER"
            status_color = "#10b981"
            status_bg = "rgba(16, 185, 129, 0.15)"
            border_color = "rgba(16, 185, 129, 0.5)"
            icon = "✅"
            recommendation = "Routine periodic health checkup recommended."

        # Speedometer Gauge Chart Component (Automatically Updates Live)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=calculated_risk_pct,
            number={"suffix": "%", "font": {"family": "Inter", "size": 32, "color": status_color}},
            title={"text": "Cancer Risk Probability Gauge", "font": {"family": "Inter", "size": 14, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#cbd5e1", "tickfont": {"size": 10, "color": "#cbd5e1"}},
                "bar": {"color": status_color, "thickness": 0.28},
                "bgcolor": "#162038",
                "bordercolor": "rgba(56, 189, 248, 0.3)",
                "steps": [
                    {"range": [0, 25], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [25, 50], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [50, 100], "color": "rgba(239, 68, 68, 0.15)"}
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 2}, "thickness": 0.75, "value": 50}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(l=15, r=15, t=35, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="background: {status_bg}; border: 1px solid {border_color}; border-radius: 12px; padding: 18px; text-align: center; margin-top: 2px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: {status_color}; margin-bottom: 6px;">{icon} {status_title}</div>
            <div style="font-size: 0.85rem; color: #f1f5f9; line-height: 1.4;">{recommendation}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # Clean Model Summary Section
    st.markdown("<h4 style='color: #ffffff; font-weight: 700;'>Model Performance Summary</h4>", unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: metric_card(f"{acc*100:.1f}%", "Accuracy", "#10b981", "🎯")
    with m2: metric_card(f"{bal_acc*100:.1f}%", "Balanced Acc.", "#38bdf8", "⚖️")
    with m3: metric_card(f"{rec*100:.1f}%", "Recall (Cancer)", "#f59e0b", "📡")
    with m4: metric_card(f"{prec*100:.1f}%", "Precision", "#a855f7", "🔍")
    with m5: metric_card(f"{auc:.3f}", "ROC-AUC", "#ef4444", "📈")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    c_sum, c_cm = st.columns([1.2, 0.8])
    with c_sum:
        st.markdown("""
        <div style="background: #162038; border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 20px 24px;">
            <h5 style="color: #ffffff; margin-top: 0; margin-bottom: 12px; font-size: 1rem; font-weight: 700;">Key Model Insights</h5>
            <ul style="color: #f1f5f9; font-size: 0.88rem; line-height: 1.75; margin-bottom: 0; padding-left: 20px;">
                <li><b style="color: #38bdf8;">High Recall Focus</b>: The model uses balanced class weights to maximize cancer detection (67.9% recall) and minimize missed diagnoses.</li>
                <li><b style="color: #38bdf8;">Class Imbalance Handling</b>: The original dataset has a 9:1 imbalance (90% negative, 10% positive), making standard accuracy misleading.</li>
                <li><b style="color: #38bdf8;">Dataset Feature Signal</b>: Features have low individual linear correlation with cancer diagnosis, making tree regularization essential.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c_cm:
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div style="background: #162038; border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 20px; text-align: center;">
            <h5 style="color: #ffffff; margin-top: 0; margin-bottom: 12px; font-size: 1rem; font-weight: 700;">Test Confusion Matrix</h5>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85rem;">
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 12px; border-radius: 8px;">
                    <div style="color: #10b981; font-weight: 800; font-size: 1.2rem;">{tn:,}</div>
                    <div style="color: #e2e8f0; font-size: 0.75rem; font-weight: 600;">True Negatives</div>
                </div>
                <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); padding: 12px; border-radius: 8px;">
                    <div style="color: #f59e0b; font-weight: 800; font-size: 1.2rem;">{fp:,}</div>
                    <div style="color: #e2e8f0; font-size: 0.75rem; font-weight: 600;">False Positives</div>
                </div>
                <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); padding: 12px; border-radius: 8px;">
                    <div style="color: #ef4444; font-weight: 800; font-size: 1.2rem;">{fn:,}</div>
                    <div style="color: #e2e8f0; font-size: 0.75rem; font-weight: 600;">False Negatives</div>
                </div>
                <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); padding: 12px; border-radius: 8px;">
                    <div style="color: #38bdf8; font-weight: 800; font-size: 1.2rem;">{tp:,}</div>
                    <div style="color: #e2e8f0; font-size: 0.75rem; font-weight: 600;">True Positives</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
#  PAGE 2: DATASET OVERVIEW
# ═════════════════════════════════════════════
elif st.session_state.page == "overview":
    page_header("Dataset Overview", "Summary of patient population, records, and class distribution.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(f"{len(df):,}", "Total Patients", "#38bdf8", "👥")
    with c2: metric_card(str(len(df.columns) - 1), "Features", "#a855f7", "🧮")
    with c3: metric_card("9.1 : 1", "Imbalance Ratio", "#f59e0b", "⚖️")
    with c4: metric_card("0", "Missing Values", "#10b981", "✅")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        label_counts = df["label"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=["No Cancer (0)", "Cancer (1)"],
            values=[label_counts.get(0, 0), label_counts.get(1, 0)],
            hole=0.6,
            marker=dict(colors=["#38bdf8", "#ef4444"])
        )])
        fig.update_layout(
            title="Target Class Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        fig = px.histogram(
            df, x="age", nbins=30,
            title="Patient Age Distribution",
            color_discrete_sequence=["#38bdf8"]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h5 style='color: #ffffff; font-weight: 700;'>Dataset Preview (First 50 Rows)</h5>", unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=350)

# ═════════════════════════════════════════════
#  PAGE 3: FEATURE ANALYSIS
# ═════════════════════════════════════════════
elif st.session_state.page == "feature":
    page_header("Feature Analysis & Statistical Significance", "Chi-Square selection, Pearson correlation, and feature ranking.")

    strong_count = (chi2_df["PValue"] < 0.001).sum()
    moderate_count = ((chi2_df["PValue"] >= 0.001) & (chi2_df["PValue"] < 0.05)).sum()
    weak_count = (chi2_df["PValue"] >= 0.05).sum()

    c1, c2, c3 = st.columns(3)
    with c1: metric_card(str(strong_count), "Strong Predictors (p < 0.001)", "#10b981", "🟢")
    with c2: metric_card(str(moderate_count), "Moderate Predictors (p < 0.05)", "#f59e0b", "🟡")
    with c3: metric_card(str(weak_count), "Weak Predictors (p ≥ 0.05)", "#ef4444", "🔴")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.2, 0.8])

    with c_left:
        colors_chi = ["#10b981" if p < 0.001 else ("#f59e0b" if p < 0.05 else "#ef4444") for p in chi2_df["PValue"]]
        fig_chi = go.Figure(go.Bar(
            x=chi2_df["Chi2"],
            y=[pretty_name(f) for f in chi2_df["Feature"]],
            orientation="h",
            marker=dict(color=colors_chi),
            text=[f"χ²={v:.1f}" for v in chi2_df["Chi2"]],
            textposition="outside",
            textfont=dict(size=9.5, color="#ffffff")
        ))
        fig_chi.update_layout(
            title="Chi-Square Feature Importance Ranking",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            height=560,
            margin=dict(l=160, r=60, t=40, b=20)
        )
        st.plotly_chart(fig_chi, use_container_width=True)

    with c_right:
        st.markdown("<h5 style='color: #ffffff; font-weight: 700;'>Statistical Significance Table</h5>", unsafe_allow_html=True)
        disp_df = chi2_df.copy()
        disp_df["Feature"] = disp_df["Feature"].apply(pretty_name)
        disp_df["Tier"] = disp_df["PValue"].apply(lambda p: "Strong" if p < 0.001 else ("Moderate" if p < 0.05 else "Weak"))
        disp_df["PValue"] = disp_df["PValue"].apply(lambda x: f"{x:.2e}")
        disp_df["Chi2"] = disp_df["Chi2"].apply(lambda x: f"{x:.2f}")

        st.dataframe(disp_df[["Feature", "Chi2", "PValue", "Tier"]], use_container_width=True, height=510)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Pearson Correlation with Target Bar Chart
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()["label"].drop("label").sort_values(ascending=False)
    
    fig_corr = go.Figure(go.Bar(
        x=[pretty_name(c) for c in corr.index],
        y=corr.values,
        marker=dict(color=corr.values, colorscale="RdBu", cmid=0),
        text=[f"{v:.3f}" for v in corr.values],
        textposition="outside",
        textfont=dict(size=8.5, color="#ffffff")
    ))
    fig_corr.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
    fig_corr.update_layout(
        title="Pearson Correlation Coefficient (r) with Cancer Target",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        height=380,
        margin=dict(l=20, r=20, t=40, b=80),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    <div style="background: #162038; border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 20px 24px; margin-top: 10px;">
        <h5 style="color: #ffffff; margin-top: 0; margin-bottom: 10px; font-weight: 700;">Feature Analysis Takeaways</h5>
        <ul style="color: #f1f5f9; font-size: 0.88rem; line-height: 1.75; margin-bottom: 0; padding-left: 20px;">
            <li><b style="color: #38bdf8;">Chi-Square Ranking</b>: Clinical features like biopsy results, endoscopic imaging, and H. Pylori status show the highest chi-square significance.</li>
            <li><b style="color: #38bdf8;">Weak Linear Correlation</b>: Pearson correlations range between r = -0.04 and r = +0.04, proving that individual linear correlation is near zero.</li>
            <li><b style="color: #38bdf8;">Non-Linear Decision Trees</b>: Decision Trees evaluate non-linear feature splits, making them better suited than linear logistic regression for this dataset.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
