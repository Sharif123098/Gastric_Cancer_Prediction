import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from sklearn.feature_selection import chi2
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GCS · Gastric Cancer Study",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');

html, body, .stApp { background:#040c18 !important; }
.main .block-container { padding: 2rem 2.5rem 4rem; max-width:1400px; }
#MainMenu,footer,header { visibility:hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#060f1e 0%,#04090f 100%) !important;
    border-right: 1px solid rgba(0,245,255,.12);
}
[data-testid="stSidebar"] > div:first-child { padding:0; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#04090f; }
::-webkit-scrollbar-thumb { background:rgba(0,245,255,.25); border-radius:3px; }

.stButton>button {
    width:100%;
    background:transparent;
    border:1px solid rgba(0,245,255,.12);
    color:rgba(160,210,240,.55);
    font-family:'IBM Plex Mono',monospace;
    font-size:.75rem;
    letter-spacing:.1em;
    text-transform:uppercase;
    padding:13px 18px;
    border-radius:8px;
    transition:all .25s;
    text-align:left;
}
.stButton>button:hover {
    background:rgba(0,245,255,.07);
    border-color:rgba(0,245,255,.35);
    color:#00f5ff;
    box-shadow:0 0 18px rgba(0,245,255,.12);
    transform:translateX(3px);
}
.stButton>button:focus { box-shadow:0 0 0 2px rgba(0,245,255,.3) !important; }

[data-testid="stDataFrame"] iframe { border-radius:10px; }
.stSlider [data-baseweb="slider"] { padding-top:6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PALETTE + THEME HELPER
# ─────────────────────────────────────────────
CYAN, RED, GREEN, ORANGE, PURPLE, PINK = "#00f5ff", "#ff3a5c", "#00ff99", "#ffb347", "#b66dff", "#ff6eb4"
PALETTE = [CYAN, RED, GREEN, ORANGE, PURPLE, PINK,
           "#4df0ff", "#ff8fa3", "#6dffb4", "#ffd06d"]


def apply_theme(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Orbitron",
                   size=14, color="#e8f4ff"), x=0.03, y=0.97),
        plot_bgcolor="rgba(5,12,22,0)",
        paper_bgcolor="rgba(6,15,28,.7)",
        font=dict(family="IBM Plex Mono", color="#9abcd6", size=11),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,245,255,.15)",
                    borderwidth=1, font=dict(size=10)),
        xaxis=dict(gridcolor="rgba(0,245,255,.07)",
                   linecolor="rgba(0,245,255,.15)", zerolinecolor="rgba(0,245,255,.1)"),
        yaxis=dict(gridcolor="rgba(0,245,255,.07)",
                   linecolor="rgba(0,245,255,.15)", zerolinecolor="rgba(0,245,255,.1)"),
        colorway=PALETTE,
    )
    return fig


def page_header(tag, title, subtitle):
    st.markdown(f"""
    <div style="margin-bottom:2rem;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:rgba(0,245,255,.5);
                  letter-spacing:.25em;text-transform:uppercase;margin-bottom:6px;">{tag}</div>
      <div style="font-family:'Orbitron',monospace;font-size:1.9rem;font-weight:700;color:#e8f4ff;
                  letter-spacing:.04em;line-height:1.2;">{title}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:rgba(0,245,255,.65);
                  margin-top:8px;letter-spacing:.04em;">{subtitle}</div>
      <div style="height:1px;background:linear-gradient(90deg,rgba(0,245,255,.4),transparent);
                  margin-top:20px;"></div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(value, label, color=CYAN, icon=""):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,245,255,.04) 0%,rgba(4,8,16,.85) 100%);
                border:1px solid rgba(0,245,255,.18);border-radius:12px;padding:22px 18px;
                text-align:center;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,transparent,{color},transparent);"></div>
      <div style="font-size:1.3rem;margin-bottom:4px;">{icon}</div>
      <div style="font-family:'Orbitron',monospace;font-size:1.85rem;font-weight:700;
                  color:{color};text-shadow:0 0 22px {color}80;">{value}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:rgba(160,210,240,.55);
                  text-transform:uppercase;letter-spacing:.12em;margin-top:6px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def info_box(text, kind="info"):
    colors = {"info": (CYAN, "rgba(0,245,255,.05)"), "warn": (ORANGE, "rgba(255,179,71,.05)"),
              "success": (GREEN, "rgba(0,255,153,.05)"), "danger": (RED, "rgba(255,58,92,.05)")}
    c, bg = colors.get(kind, colors["info"])
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {c}33;border-left:3px solid {c};
                border-radius:0 8px 8px 0;padding:14px 18px;font-family:'IBM Plex Mono',monospace;
                font-size:.82rem;color:rgba(210,235,255,.85);margin:12px 0;line-height:1.6;">
      {text}
    </div>
    """, unsafe_allow_html=True)


def section_divider():
    st.markdown("""<div style="height:1px;background:linear-gradient(90deg,
        transparent,rgba(0,245,255,.2),transparent);margin:28px 0;"></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FEATURE METADATA — matches cleaned_gcs_kushal.csv schema exactly
# ─────────────────────────────────────────────
ONEHOT_PREFIXES = {
    "existing_conditions_": "Existing Condition",
    "mature_mirna_id_": "miRNA Identifier",
    "target_symbol_": "Target Gene Symbol",
}
LABEL_OVERRIDES = {
    "age": "Age (years)", "gender": "Gender", "family_history": "Family History of Cancer",
    "smoking_habits": "Smoking Habits", "alcohol_consumption": "Alcohol Consumption",
    "helicobacter_pylori_infection": "H. Pylori Infection", "dietary_habits": "Dietary Habits (Salt Intake)",
    "endoscopic_images": "Endoscopic Imaging", "biopsy_results": "Biopsy Result", "ct_scan": "CT Scan",
    "diana_microt": "DIANA-microT", "elmmo": "ElMMo", "microcosm": "MicroCosm", "miranda": "miRanda",
    "mirdb": "miRDB", "pictar": "PicTar", "pita": "PITA", "targetscan": "TargetScan",
    "predicted.sum": "Predicted-Tool Agreement", "all.sum": "Total Algorithm Score",
}
# Exact binary mappings as confirmed
CLINICAL_BINARY_LABELS = {
    "gender": ("Male", "Female"),
    "dietary_habits": ("High_Salt", "Low_Salt"),
    "endoscopic_images": ("Abnormal", "Normal"),
    "biopsy_results": ("Positive", "Negative"),
    "ct_scan": ("Positive", "Negative"),
    "family_history": ("Yes", "No"),
    "smoking_habits": ("Yes", "No"),
    "alcohol_consumption": ("Yes", "No"),
    "helicobacter_pylori_infection": ("Yes", "No"),
}
CLINICAL_BINARY_COLS = list(CLINICAL_BINARY_LABELS.keys())


def pretty_label(col):
    return LABEL_OVERRIDES.get(col, col.replace("_", " ").replace(".", " ").title())


def detect_onehot_groups(columns):
    groups = {}
    for prefix, group_label in ONEHOT_PREFIXES.items():
        cols = [c for c in columns if c.startswith(prefix)]
        if cols:
            groups[group_label] = cols
    return groups

# ─────────────────────────────────────────────
#  DATA + MODEL (automatic; no user upload required)
# ─────────────────────────────────────────────


APP_DIR = Path(__file__).resolve().parent

# The notebook trains on this exact cleaned dataset:
# ../Dataset/cleaned_gcs_kushal.csv
# We also accept the same file beside the Streamlit app for easier deployment.
DATASET_CANDIDATES = [
    APP_DIR / "cleaned_gcs_kushal.csv",
    APP_DIR / "Dataset" / "cleaned_gcs_kushal.csv",
    APP_DIR.parent / "Dataset" / "cleaned_gcs_kushal.csv",
]

MODEL_MAX_DEPTH = 8
MODEL_MIN_SAMPLES_LEAF = 50
MODEL_TEST_SIZE = 0.20


def find_training_dataset():
    for candidate in DATASET_CANDIDATES:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"• {p}" for p in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "The cleaned training dataset could not be found. "
        "Place 'cleaned_gcs_kushal.csv' in the same folder as this app "
        "or in a 'Dataset' subfolder.\n\n"
        f"Searched locations:\n{searched}"
    )


@st.cache_data(show_spinner=False)
def load_training_data():
    path = find_training_dataset()
    df = pd.read_csv(path)

    # Match the notebook exactly: boolean one-hot columns -> integer 0/1.
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)

    if "label" not in df.columns:
        raise ValueError(
            "The training dataset must contain the target column 'label'."
        )

    # The model expects a fully numeric feature matrix.
    feature_df = df.drop(columns=["label"])
    non_numeric = feature_df.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise ValueError(
            "The cleaned dataset still contains non-numeric feature columns: "
            + ", ".join(non_numeric)
        )

    return df


@st.cache_resource(show_spinner=False)
def train_final_model(df):
    # This reproduces the notebook's final model configuration exactly:
    # DecisionTreeClassifier(class_weight='balanced', max_depth=8,
    #                        min_samples_leaf=50, random_state=42)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=MODEL_TEST_SIZE, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
    )

    model = DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=MODEL_MAX_DEPTH,
        min_samples_leaf=MODEL_MIN_SAMPLES_LEAF,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return (
        model,
        X.columns.tolist(),
        X,
        y,
        X_train,
        y_train,
        y_test,
        y_pred,
        y_prob,
    )


@st.cache_data(show_spinner=False)
def compute_chi2(df, target_col="label"):
    X = df.drop(columns=[target_col]).clip(lower=0)
    y = df[target_col]
    scores, pvals = chi2(X, y)
    return pd.DataFrame(
        {"Feature": X.columns, "Chi2": scores, "PValue": pvals}
    ).sort_values("Chi2", ascending=False)


# ─────────────────────────────────────────────
#  SIDEBAR — data source + model console + nav
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    # Open directly on the model/prediction page so users can enter patient values immediately.
    st.session_state.page = "model"

NAV = [
    ("overview", "📁", "Dataset Overview"),
    ("encoding", "🔢", "Feature Encoding"),
    ("eda", "📊", "EDA"),
    ("feature", "🔬", "Feature Analysis"),
    ("hypothesis", "🧪", "Hypothesis"),
    ("model", "🤖", "Model"),
]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:32px 20px 28px;border-bottom:1px solid rgba(0,245,255,.1);margin-bottom:20px;">
      <div style="font-size:2.2rem;margin-bottom:8px;">🧬</div>
      <div style="font-family:'Orbitron',monospace;font-size:1.15rem;font-weight:900;color:#00f5ff;
                  letter-spacing:.18em;text-shadow:0 0 28px rgba(0,245,255,.55);">GCS</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:rgba(160,210,240,.4);
                  letter-spacing:.22em;text-transform:uppercase;margin-top:5px;">Gastric Cancer Study</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:rgba(0,245,255,.35);
        letter-spacing:.22em;text-transform:uppercase;padding:0 4px;margin:16px 0 8px;">Model Console</div>""",
                unsafe_allow_html=True)
    st.caption("")
    metric_card("DT", "Decision Tree", CYAN, "🌳")
    # st.caption(
    # "Balanced class weights · max_depth=8 · min_samples_leaf=50 · test_size=20%")

    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:rgba(0,245,255,.35);
        letter-spacing:.22em;text-transform:uppercase;padding:0 4px;margin:20px 0 8px;">Navigation</div>""",
                unsafe_allow_html=True)

    for key, icon, label in NAV:
        is_active = st.session_state.page == key
        if is_active:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:12px 18px;
                        background:rgba(0,245,255,.09);border:1px solid rgba(0,245,255,.42);
                        border-radius:8px;margin:3px 0;cursor:default;box-shadow:0 0 20px rgba(0,245,255,.12);">
              <span style="font-size:1rem;">{icon}</span>
              <span style="font-family:'IBM Plex Mono',monospace;font-size:.75rem;letter-spacing:.08em;
                           text-transform:uppercase;color:#00f5ff;">{label}</span>
              <span style="margin-left:auto;width:6px;height:6px;border-radius:50%;background:#00f5ff;
                           box-shadow:0 0 8px #00f5ff;"></span>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(f"{icon}  {label}", key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()

    st.markdown("""
    <div style="position:fixed;bottom:0;left:0;width:260px;padding:16px 20px;
                border-top:1px solid rgba(0,245,255,.08);background:#04090f;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:rgba(160,210,240,.3);
                  letter-spacing:.1em;text-align:center;">GASTRIC CANCER STUDY · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD + TRAIN AUTOMATICALLY
# ─────────────────────────────────────────────
if "model" not in st.session_state:
    try:
        with st.spinner("Loading the notebook dataset and initializing the trained Decision Tree..."):
            df = load_training_data()

            (
                model, feature_cols, X_full, y_full, X_train, y_train,
                y_test, y_pred, y_prob
            ) = train_final_model(df)

            chi2_df = compute_chi2(df)

            st.session_state.update(
                dict(
                    df=df,
                    model=model,
                    feature_cols=feature_cols,
                    X_full=X_full,
                    y_full=y_full,
                    X_train=X_train,
                    y_train=y_train,
                    y_test=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    chi2_df=chi2_df,
                )
            )
    except Exception as e:
        st.error("The app could not initialize the model.")
        st.exception(e)
        st.stop()

df = st.session_state.df
model = st.session_state.model
feature_cols = st.session_state.feature_cols
X_full = st.session_state.X_full
y_full = st.session_state.y_full
X_train = st.session_state.X_train
y_train = st.session_state.y_train
y_test = st.session_state.y_test
y_pred = st.session_state.y_pred
y_prob = st.session_state.y_prob
chi2_df = st.session_state.chi2_df

onehot_groups = detect_onehot_groups(feature_cols)
onehot_cols_flat = {c for cols in onehot_groups.values() for c in cols}
plain_cols = [c for c in feature_cols if c not in onehot_cols_flat]
clinical_cols = [
    c for c in plain_cols if c in CLINICAL_BINARY_COLS or c == "age"]
score_cols = [c for c in plain_cols if c not in clinical_cols]

# ═════════════════════════════════════════════
#  PAGE 1 · DATASET OVERVIEW
# ═════════════════════════════════════════════
if st.session_state.page == "overview":
    page_header("01 / 06", "DATASET OVERVIEW",
                "Patient-level gastric cancer dataset — shape, quality & target balance")

    n_dupes = int(df.duplicated().sum())
    n_missing = int(df.isna().sum().sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"{len(df):,}", "Total Records", CYAN, "📋")
    with c2:
        metric_card(str(len(df.columns) - 1), "Features", PURPLE, "🧮")
    with c3:
        metric_card(str(n_dupes), "Duplicate Rows", ORANGE, "⚠️")
    with c4:
        metric_card(str(n_missing), "Missing Values",
                    GREEN if n_missing == 0 else RED, "✅")

    section_divider()
    c_left, c_right = st.columns(2)

    with c_left:
        label_counts = df["label"].value_counts().reindex([1, 0], fill_value=0)
        fig = go.Figure(data=[go.Pie(
            labels=["Cancer Positive", "Cancer Negative"], values=label_counts.values, hole=.62,
            marker=dict(colors=[RED, CYAN], line=dict(
                color="#04090f", width=3)),
            textinfo="percent+label", textfont=dict(family="IBM Plex Mono", size=11, color="#e8f4ff"),
            hovertemplate="%{label}<br>Count: %{value}<br>Pct: %{percent}<extra></extra>",
        )])
        fig.add_annotation(text=f"<b>{len(df)}</b><br><span style='font-size:10px'>Records</span>",
                           x=.5, y=.5, font=dict(family="Orbitron", size=16, color="#00f5ff"), showarrow=False)
        fig = apply_theme(fig, "🎯 Target Distribution", 380)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        if "gender" in df.columns:
            male, female = int((df["gender"] == 1).sum()), int(
                (df["gender"] == 0).sum())
            fig = go.Figure(data=[go.Bar(
                x=["Male", "Female"], y=[male, female],
                marker=dict(color=[CYAN, PINK]),
                text=[male, female], textposition="outside",
                textfont=dict(family="IBM Plex Mono",
                              size=12, color="#e8f4ff"),
                hovertemplate="%{x}: %{y}<extra></extra>",
            )])
            fig = apply_theme(fig, "👥 Gender Distribution", 380)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    section_divider()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["age"], nbinsx=30, marker=dict(color=CYAN, opacity=.75,
                  line=dict(color="rgba(0,245,255,.3)", width=1)), name="Age",
        hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>"))
    fig.add_vline(x=df["age"].mean(), line_dash="dash", line_color=ORANGE,
                  annotation_text=f"Mean: {df['age'].mean():.1f}",
                  annotation_font=dict(family="IBM Plex Mono", color=ORANGE, size=11))
    fig = apply_theme(fig, "📈 Age Distribution of Patients", 320)
    st.plotly_chart(fig, use_container_width=True)

    section_divider()
    pos_pct = 100 * label_counts[1] / len(df)
    info_box(f"📌 Target prevalence: <b>{pos_pct:.1f}%</b> cancer-positive vs <b>{100-pos_pct:.1f}%</b> negative "
             f"— roughly a <b>{label_counts[0]/max(label_counts[1], 1):.1f}:1</b> imbalance. "
             f"F1 / recall / precision on the minority class matter far more here than raw accuracy.", "warn")

    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:rgba(0,245,255,.6);
        text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;">📄 Raw Data Preview</div>""",
                unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=360)

# ═════════════════════════════════════════════
#  PAGE 2 · FEATURE ENCODING
# ═════════════════════════════════════════════
elif st.session_state.page == "encoding":
    page_header("02 / 06", "FEATURE ENCODING & CORRELATIONS",
                "Binary mappings, one-hot groups & inter-feature correlation")

    label_counts = df["label"].value_counts().reindex([1, 0], fill_value=0)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(str(int(label_counts[1])), "Cancer Positive", RED, "🔴")
    with c2:
        metric_card(str(int(label_counts[0])), "Cancer Negative", CYAN, "🔵")
    with c3:
        metric_card(
            f"{label_counts[0]/max(label_counts[1], 1):.1f}:1", "Imbalance Ratio", ORANGE, "⚖️")
    with c4:
        metric_card(
            f"{100*label_counts[1]/len(df):.1f}%", "Minority Prevalence", PURPLE, "📊")

    section_divider()
    c_left, c_right = st.columns([1.1, 1])

    with c_left:
        rows = [("Cancer +", 1, "label"), ("Cancer -", 0, "label")]
        for col, (pos_lbl, neg_lbl) in CLINICAL_BINARY_LABELS.items():
            rows.append((pos_lbl, 1, col))
            rows.append((neg_lbl, 0, col))
        for group_label, cols in onehot_groups.items():
            prefix = [p for p in ONEHOT_PREFIXES if ONEHOT_PREFIXES[p]
                      == group_label][0]
            for c in cols:
                rows.append(
                    (c.split(prefix, 1)[-1].replace("_", " "), 1, group_label + " (one-hot)"))
        enc_df = pd.DataFrame(
            rows, columns=["Original Value", "Encoded Value", "Column"])
        fig = go.Figure(data=[go.Table(
            header=dict(values=["<b>Value</b>", "<b>Encoded</b>", "<b>Column</b>"],
                        fill_color="rgba(0,245,255,.12)", align="left",
                        font=dict(family="IBM Plex Mono",
                                  size=11, color="#00f5ff"),
                        line_color="rgba(0,245,255,.2)", height=32),
            cells=dict(values=[enc_df["Original Value"], enc_df["Encoded Value"], enc_df["Column"]],
                       fill_color=["rgba(5,12,25,.7)"], align="left",
                       font=dict(family="IBM Plex Mono",
                                 size=10, color="#cce8ff"),
                       line_color="rgba(0,245,255,.1)", height=28)
        )])
        fig = apply_theme(fig, "🔢 Binary & One-Hot Encoding Map", 460)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        info_box("✅ Boolean one-hot columns are cast to int(0/1) on load. Binary clinical flags follow the "
                 "confirmed mapping: <b>gender</b> (Male=1/Female=0), <b>dietary_habits</b> (High_Salt=1/"
                 "Low_Salt=0), <b>endoscopic_images</b> (Abnormal=1/Normal=0), <b>biopsy_results</b> and "
                 "<b>ct_scan</b> (Positive=1/Negative=0).", "success")
        info_box("🩺 Remaining Yes/No clinical flags — <b>family_history</b>, <b>smoking_habits</b>, "
                 "<b>alcohol_consumption</b>, <b>helicobacter_pylori_infection</b> — follow the standard "
                 "1=Yes / 0=No convention.", "info")
        info_box("🧬 Three feature families are one-hot encoded: <b>existing_conditions_*</b> (Chronic "
                 "Gastritis / Diabetes / Unknown), <b>mature_mirna_id_*</b> (MIR123_1 / MIR234_2 / MIR345_3), "
                 "and <b>target_symbol_*</b> (CDH1 / KRAS / TP53) — each row has exactly one '1' within its group.",
                 "info")

    section_divider()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale=[[0, "#1a0a2e"], [0.5, "#04090f"], [1, "#00f5ff"]],
        text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
        textfont=dict(size=7, family="IBM Plex Mono"),
        hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>", zmin=-1, zmax=1,
    ))
    fig = apply_theme(fig, "🔥 Correlation Heatmap — All Numeric Features", 560)
    fig.update_layout(margin=dict(l=140, r=20, t=50, b=140))
    st.plotly_chart(fig, use_container_width=True)

    section_divider()
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:rgba(0,245,255,.6);
        text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;">📄 Cleaned Data Preview</div>""",
                unsafe_allow_html=True)
    st.dataframe(df.head(50).style.background_gradient(cmap="Blues", subset=["age"]),
                 use_container_width=True, height=360)

# ═════════════════════════════════════════════
#  PAGE 3 · EDA
# ═════════════════════════════════════════════
elif st.session_state.page == "eda":
    page_header("03 / 06", "EXPLORATORY DATA ANALYSIS",
                "Distributions & relationships between features and the target")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"{df['age'].mean():.1f}", "Mean Age", CYAN, "👤")
    with c2:
        metric_card(f"{df['age'].std():.1f}", "Age Std Dev", PURPLE, "📐")
    with c3:
        metric_card(str(int(df["age"].max())), "Max Age", ORANGE, "📅")
    with c4:
        metric_card(str(int(df["age"].min())), "Min Age", GREEN, "📅")

    section_divider()
    c_left, c_right = st.columns(2)
    with c_left:
        fig = go.Figure()
        for val, lbl, color in [(1, "Cancer Positive", RED), (0, "Cancer Negative", CYAN)]:
            subset = df[df["label"] == val]["age"]
            fig.add_trace(go.Histogram(x=subset, name=lbl, nbinsx=20, marker_color=color, opacity=.65,
                          hovertemplate=f"{lbl}<br>Age: %{{x}}<br>Count: %{{y}}<extra></extra>"))
        fig = apply_theme(fig, "📊 Age Distribution by Cancer Status", 380)
        fig.update_layout(barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        bins = list(range(int(df["age"].min() // 10 * 10),
                    int(df["age"].max() // 10 * 10) + 20, 10))
        age_groups = pd.cut(df["age"], bins=bins)
        group_df = pd.DataFrame(
            {"age_group": age_groups.astype(str), "label": df["label"]})
        pivot = group_df.groupby(
            ["age_group", "label"]).size().reset_index(name="count")
        pivot["status"] = pivot["label"].map({1: "Positive", 0: "Negative"})
        fig = px.bar(pivot, x="age_group", y="count", color="status",
                     color_discrete_map={"Positive": RED, "Negative": CYAN}, barmode="group",
                     labels={"age_group": "Age Group", "count": "Patients"})
        fig = apply_theme(fig, "📊 Cancer Cases by Age Group", 380)
        st.plotly_chart(fig, use_container_width=True)

    section_divider()
    clinical_present = [c for c in CLINICAL_BINARY_COLS if c in df.columns]
    prevalence = df[clinical_present].mean().sort_values()
    fig = go.Figure(go.Bar(
        x=prevalence.values * 100, y=[pretty_label(c) for c in prevalence.index], orientation="h",
        marker=dict(color=prevalence.values, colorscale=[
                    [0, PURPLE], [0.5, CYAN], [1, GREEN]]),
        text=[f"{v*100:.1f}%" for v in prevalence.values], textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color="#e8f4ff"),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig = apply_theme(fig, "🔍 Clinical Flag Prevalence (%)", 420)
    fig.update_layout(margin=dict(l=180, r=60, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section_divider()
    st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:600;
        color:#e8f4ff;letter-spacing:.06em;margin-bottom:16px;">
        🔥 Feature Prevalence by Cancer Status</div>""", unsafe_allow_html=True)

    heat_rows = []
    for c in clinical_present:
        pos_rate = df[df["label"] == 1][c].mean()
        neg_rate = df[df["label"] == 0][c].mean()
        heat_rows.append({"Feature": pretty_label(c), "Positive": pos_rate, "Negative": neg_rate,
                          "Difference": pos_rate - neg_rate})
    heat_df = pd.DataFrame(heat_rows).sort_values(
        "Difference", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=heat_df["Difference"], y=heat_df["Feature"], orientation="h",
            marker=dict(color=heat_df["Difference"], colorscale=[
                        [0, PURPLE], [0.4, "#04090f"], [1, RED]]),
            hovertemplate="%{y}<br>Δ Prevalence: %{x:.3f}<extra></extra>",
        ))
        fig = apply_theme(fig, "📈 Feature: Positive - Negative Rate", 400)
        fig.update_layout(margin=dict(l=180, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Cancer +", x=heat_df["Feature"], y=heat_df["Positive"], marker_color=RED, opacity=.8))
        fig.add_trace(go.Bar(
            name="Cancer -", x=heat_df["Feature"], y=heat_df["Negative"], marker_color=CYAN, opacity=.8))
        fig = apply_theme(fig, "📊 Feature Rate: +ve vs -ve Patients", 400)
        fig.update_layout(barmode="group", xaxis_tickangle=-
                          40, xaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    section_divider()
    top_feat = heat_df.iloc[0]
    info_box(f"🔎 <b>Key EDA Insight:</b> <b>{top_feat['Feature']}</b> shows the largest gap between cancer-positive "
             f"and cancer-negative patients (Δ = {top_feat['Difference']:.3f}). But note: as the Model page shows, "
             f"this dataset's features carry almost no real predictive signal for the label overall — the "
             f"gaps here are small in absolute terms.", "warn")

# ═════════════════════════════════════════════
#  PAGE 4 · FEATURE ANALYSIS
# ═════════════════════════════════════════════
elif st.session_state.page == "feature":
    page_header("04 / 06", "FEATURE ANALYSIS",
                "Chi-Square selection · Feature importance · Correlation with target")

    def get_tier(pv):
        if pv < 0.001:
            return ("🟢 Strong", GREEN)
        elif pv < 0.05:
            return ("🟡 Moderate", ORANGE)
        else:
            return ("🔴 Weak", RED)

    strong = (chi2_df["PValue"] < 0.001).sum()
    moderate = ((chi2_df["PValue"] >= 0.001) &
                (chi2_df["PValue"] < 0.05)).sum()
    weak = (chi2_df["PValue"] >= 0.05).sum()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(str(strong), "Strong Predictors (p<0.001)", GREEN, "🟢")
    with c2:
        metric_card(str(moderate), "Moderate Predictors", ORANGE, "🟡")
    with c3:
        metric_card(str(weak), "Weak Predictors", RED, "🔴")

    section_divider()
    c_left, c_right = st.columns([3, 2])
    with c_left:
        colors_chi = [GREEN if p < 0.001 else ORANGE if p <
                      0.05 else RED for p in chi2_df["PValue"]]
        fig = go.Figure(go.Bar(
            x=chi2_df["Chi2"], y=[pretty_label(f) for f in chi2_df["Feature"]], orientation="h",
            marker=dict(color=colors_chi),
            text=[f"χ²={v:.1f}" for v in chi2_df["Chi2"]], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=8, color="#e8f4ff"),
            hovertemplate="%{y}<br>χ² = %{x:.2f}<extra></extra>",
        ))
        fig = apply_theme(fig, "📊 Chi-Square Feature Importance", 620)
        fig.update_layout(margin=dict(l=190, r=80, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c_right:
        disp = chi2_df.copy()
        disp["Tier"] = disp["PValue"].apply(lambda p: get_tier(p)[0])
        disp["Feature"] = disp["Feature"].apply(pretty_label)
        disp["PValue"] = disp["PValue"].apply(lambda x: f"{x:.2e}")
        disp["Chi2"] = disp["Chi2"].apply(lambda x: f"{x:.2f}")
        st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:rgba(0,245,255,.6);
            text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;">Chi-Square Results</div>""",
                    unsafe_allow_html=True)
        st.dataframe(disp[["Feature", "Chi2", "PValue", "Tier"]],
                     use_container_width=True, height=580)

    section_divider()
    corr = df.select_dtypes(include=[np.number]).corr(
    )["label"].drop("label").sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=[pretty_label(c) for c in corr.index], y=corr.values,
        marker=dict(color=corr.values, colorscale=[
                    [0, RED], [0.5, "#04090f"], [1, GREEN]]),
        text=[f"{v:.3f}" for v in corr.values], textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=8, color="#e8f4ff"),
        hovertemplate="%{x}<br>r = %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,.2)")
    fig = apply_theme(fig, "📈 Pearson Correlation with Cancer Label", 380)
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    section_divider()
    top3 = chi2_df.head(3)
    max_abs_corr = corr.abs().max()
    info_box(f"<b>Top Predictors by Chi-Square:</b><br>🥇 {pretty_label(top3.iloc[0]['Feature'])} "
             f"(χ²={top3.iloc[0]['Chi2']:.1f}) &nbsp;|&nbsp; 🥈 {pretty_label(top3.iloc[1]['Feature'])} "
             f"(χ²={top3.iloc[1]['Chi2']:.1f}) &nbsp;|&nbsp; 🥉 {pretty_label(top3.iloc[2]['Feature'])} "
             f"(χ²={top3.iloc[2]['Chi2']:.1f})", "success")
    info_box(f"⚠️ Even the strongest linear correlation with the label is only r ≈ {max_abs_corr:.3f} — "
             f"a near-zero relationship. Statistical significance (low p-value) here mainly reflects the "
             f"very large sample size, not strong real-world predictive power.", "warn")

# ═════════════════════════════════════════════
#  PAGE 5 · HYPOTHESIS
# ═════════════════════════════════════════════
elif st.session_state.page == "hypothesis":
    page_header("05 / 06", "HYPOTHESIS TESTING",
                "Chi-Square significance · Class balance · Statistical validation")

    sig_mask = chi2_df["PValue"] < 0.05
    n_sig, n_insig = int(sig_mask.sum()), int((~sig_mask).sum())
    verdict = "H₀ REJECTED" if n_sig > 0 else "H₀ NOT REJECTED"
    verdict_color = GREEN if n_sig > 0 else RED

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,245,255,.04),rgba(4,8,16,.9));
                border:1px solid rgba(0,245,255,.25);border-radius:12px;padding:28px;margin-bottom:28px;">
      <div style="display:flex;gap:32px;flex-wrap:wrap;">
        <div style="flex:1;min-width:240px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:rgba(255,58,92,.7);
                      letter-spacing:.2em;text-transform:uppercase;margin-bottom:8px;">H₀ — Null Hypothesis</div>
          <div style="font-family:'DM Sans',sans-serif;font-size:.95rem;color:#cce8ff;line-height:1.6;">
            Patient/genomic features have <b>no significant association</b> with gastric cancer diagnosis.
          </div>
        </div>
        <div style="flex:1;min-width:240px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:rgba(0,255,153,.7);
                      letter-spacing:.2em;text-transform:uppercase;margin-bottom:8px;">H₁ — Alternative Hypothesis</div>
          <div style="font-family:'DM Sans',sans-serif;font-size:.95rem;color:#cce8ff;line-height:1.6;">
            At least one feature is <b>significantly associated</b> with gastric cancer diagnosis.
          </div>
        </div>
        <div style="flex:0;min-width:180px;text-align:center;padding:16px;
                    background:{verdict_color}14;border:1px solid {verdict_color}4d;border-radius:10px;">
          <div style="font-family:'Orbitron',monospace;font-size:1rem;color:{verdict_color};
                      text-shadow:0 0 18px {verdict_color}88;font-weight:700;">{verdict}</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:{verdict_color}99;
                      margin-top:6px;letter-spacing:.1em;">chi² p-values<br>(α = 0.05)</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns(2)
    with c_left:
        sig_df = chi2_df[sig_mask].sort_values("Chi2", ascending=False)
        fig = go.Figure(go.Bar(
            x=[pretty_label(f) for f in sig_df["Feature"]], y=sig_df["Chi2"],
            marker_color=GREEN, opacity=.8, name="Significant (p < 0.05)",
            text=[f"χ²={v:.1f}" for v in sig_df["Chi2"]], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=8, color="#e8f4ff"),
        ))
        fig = apply_theme(fig, "✅ Significant Features (p < 0.05)", 400)
        fig.update_layout(showlegend=False, xaxis_tickangle=-
                          35, xaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)
    with c_right:
        insig_df = chi2_df[~sig_mask].sort_values("Chi2", ascending=False)
        fig = go.Figure(go.Bar(
            x=[pretty_label(f) for f in insig_df["Feature"]], y=insig_df["Chi2"],
            marker_color=RED, opacity=.65, name="Insignificant",
            text=[f"χ²={v:.1f}" for v in insig_df["Chi2"]], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=8, color="#e8f4ff"),
        ))
        fig = apply_theme(fig, "❌ Insignificant Features (p ≥ 0.05)", 400)
        fig.update_layout(showlegend=False, xaxis_tickangle=-
                          35, xaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)

    section_divider()
    c1, c2 = st.columns(2)
    with c1:
        label_counts = df["label"].value_counts().reindex([1, 0], fill_value=0)
        pos_pct = 100 * label_counts[1] / len(df)
        fig = go.Figure(data=[go.Pie(
            labels=["Cancer +", "Cancer -"], values=label_counts.values, hole=.55,
            marker=dict(colors=[RED, CYAN], line=dict(
                color="#04090f", width=3)),
            textinfo="percent+value", textfont=dict(family="IBM Plex Mono", size=11, color="#e8f4ff"),
            pull=[.05, 0], hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
        )])
        fig.add_annotation(text=f"<b>{pos_pct:.0f}%/{100-pos_pct:.0f}%</b>", x=.5, y=.5,
                           font=dict(family="Orbitron", size=14, color=CYAN), showarrow=False)
        fig = apply_theme(fig, "⚖️ Class Distribution", 380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_p = chi2_df.sort_values("PValue").head(8)
        colors_p = [GREEN if p < 0.05 else RED for p in top_p["PValue"]]
        fig = go.Figure(go.Bar(
            x=[pretty_label(f) for f in top_p["Feature"]], y=[-np.log10(max(p, 1e-300)) for p in top_p["PValue"]],
            marker=dict(color=colors_p),
            text=[f"p={p:.2e}" for p in top_p["PValue"]], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=8, color="#e8f4ff"),
            hovertemplate="%{x}<br>-log₁₀(p): %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color=ORANGE,
                      annotation_text="p = 0.05 threshold",
                      annotation_font=dict(family="IBM Plex Mono", color=ORANGE, size=10))
        fig = apply_theme(fig, "📊 Top-8 Statistical Significance", 380)
        fig.update_layout(xaxis_tickangle=-35,
                          xaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)

    section_divider()
    if n_sig > 0:
        info_box(f"""<b>Conclusion:</b> We <b>REJECT the null hypothesis</b> (H₀) at α = 0.05.<br><br>
        {n_sig} of {len(chi2_df)} features show chi-square p-values below 0.05, giving statistical evidence
        that patient/genomic features ARE associated with gastric cancer diagnosis in this dataset.
        Class balance sits at roughly {pos_pct:.1f}% positive vs {100-pos_pct:.1f}% negative.<br><br>
        <b>Caveat:</b> statistical significance here is largely a function of the very large sample size
        ({len(df):,} rows) — the effect sizes (chi² magnitudes and correlations) are still small, and the
        trained Decision Tree's near-0.50 ROC-AUC on the test set (see Model page) shows this significance
        does not translate into real predictive power.""", "success")
    else:
        info_box("""<b>Conclusion:</b> We <b>FAIL TO REJECT the null hypothesis</b> (H₀) at α = 0.05 —
        no feature cleared the significance threshold in this sample.""", "warn")

# ═════════════════════════════════════════════
#  PAGE 6 · MODEL
# ═════════════════════════════════════════════
elif st.session_state.page == "model":

    page_header("06 / 06", "MODEL — DECISION TREE (BALANCED, HIGH RECALL)",
                f"class_weight='balanced' · max_depth={MODEL_MAX_DEPTH} · min_samples_leaf={MODEL_MIN_SAMPLES_LEAF} · "
                f"{int((1-MODEL_TEST_SIZE)*100)}/{int(MODEL_TEST_SIZE*100)} train/test split")

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1s = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card(f"{acc*100:.2f}%", "Accuracy", GREEN, "🎯")
    with c2:
        metric_card(f"{bal_acc*100:.2f}%", "Balanced Acc.", CYAN, "⚖️")
    with c3:
        metric_card(f"{prec*100:.2f}%", "Precision (cancer)", CYAN, "🔍")
    with c4:
        metric_card(f"{rec*100:.2f}%", "Recall (cancer)", ORANGE, "📡")
    with c5:
        metric_card(f"{f1s*100:.2f}%", "F1-Score (cancer)", PURPLE, "⚡")
    with c6:
        metric_card(f"{auc:.4f}", "ROC-AUC", RED, "📈")

    if auc < 0.55:
        info_box(f"⚠️ <b>Model reality check:</b> ROC-AUC of <b>{auc:.4f}</b> is essentially coin-flip "
                 f"performance (0.50 = random). Balanced accuracy of <b>{bal_acc*100:.2f}%</b> confirms the "
                 f"model isn't meaningfully separating the two classes — <code>class_weight='balanced'</code> "
                 f"is pushing recall up on the minority class, but at the cost of precision and overall "
                 f"accuracy. This dataset's features carry very little real signal for the label. Worth "
                 f"disclosing honestly rather than presenting Recall alone as a success metric.", "danger")

    section_divider()
    c_left, c_right = st.columns(2)
    with c_left:
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted Negative", "Predicted Positive"], y=["Actual Negative", "Actual Positive"],
            text=[[str(cm[i][j]) for j in range(2)] for i in range(2)], texttemplate="<b>%{text}</b>",
            textfont=dict(family="Orbitron", size=22, color="#ffffff"),
            colorscale=[[0, "rgba(4,8,16,.9)"], [0.5, "rgba(0,100,160,.4)"], [
                1, "rgba(0,245,255,.6)"]],
            showscale=False, hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
        ))
        fig = apply_theme(fig, "🔲 Confusion Matrix", 400)
        st.plotly_chart(fig, use_container_width=True)
    with c_right:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                      line=dict(color="rgba(180,180,180,.3)",
                                dash="dash", width=1.5),
                      name="Random Classifier", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=CYAN, width=3),
                      name=f"Decision Tree (AUC = {auc:.4f})", fill="tozeroy", fillcolor="rgba(0,245,255,.06)",
                      hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"))
        fig = apply_theme(fig, f"📈 ROC Curve | AUC = {auc:.4f}", 400)
        fig.update_layout(xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate", legend=dict(x=.5, y=.1))
        st.plotly_chart(fig, use_container_width=True)

    section_divider()
    c_left, c_right = st.columns(2)
    with c_left:
        prc_p, prc_r, _ = precision_recall_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prc_r, y=prc_p, mode="lines", line=dict(color=PURPLE, width=3),
                      name=f"PR Curve (AP = {ap:.4f})", fill="tozeroy", fillcolor="rgba(182,109,255,.06)",
                      hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"))
        base_rate = float(y_test.mean())
        fig.add_hline(y=base_rate, line_dash="dash", line_color="rgba(180,180,180,.4)",
                      annotation_text=f"Base rate = {base_rate:.3f}",
                      annotation_font=dict(family="IBM Plex Mono", color="rgba(200,200,200,.6)", size=10))
        fig = apply_theme(
            fig, f"📉 Precision-Recall Curve | AP = {ap:.4f}", 380)
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)
    with c_right:
        metrics_names = ["Accuracy", "Bal. Accuracy",
                         "Precision", "Recall", "F1-Score", "ROC-AUC"]
        metrics_vals = [acc, bal_acc, prec, rec, f1s, auc]
        metrics_colors = [GREEN, CYAN, CYAN, ORANGE, PURPLE, RED]
        fig = go.Figure()
        for name, score, color in zip(metrics_names, metrics_vals, metrics_colors):
            fig.add_trace(go.Bar(name=name, x=[name], y=[score*100], marker=dict(color=color, opacity=.8,
                          line=dict(color=color, width=1)), text=f"{score*100:.2f}%", textposition="outside",
                textfont=dict(family="Orbitron", size=10, color="#e8f4ff"),
                hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>"))
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,.2)", annotation_text="50% baseline",
                      annotation_font=dict(family="IBM Plex Mono", color="rgba(255,255,255,.4)", size=10))
        fig = apply_theme(fig, "🏆 Complete Model Performance Overview", 380)
        fig.update_layout(showlegend=False, yaxis=dict(range=[0, 110]))
        st.plotly_chart(fig, use_container_width=True)

    section_divider()
    tn, fp, fn, tp = cm.ravel()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(str(tp), "True Positives", GREEN, "✅")
    with c2:
        metric_card(str(tn), "True Negatives", CYAN, "✅")
    with c3:
        metric_card(str(fp), "False Positives", ORANGE, "⚠️")
    with c4:
        metric_card(str(fn), "False Negatives", RED, "❌")

    section_divider()
    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}) \
        .sort_values("Importance", ascending=False).head(15)
    fig = go.Figure(go.Bar(
        x=imp_df["Importance"], y=[pretty_label(f) for f in imp_df["Feature"]], orientation="h",
        marker=dict(color=imp_df["Importance"], colorscale=[
                    [0, PURPLE], [0.5, CYAN], [1, GREEN]]),
        hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>",
    ))
    fig = apply_theme(fig, "🌳 Top-15 Decision Tree Feature Importances", 460)
    fig.update_layout(margin=dict(l=190, r=40, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── Live Prediction Tool
    st.markdown(f"""
    <div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:700;color:#e8f4ff;
                letter-spacing:.08em;margin-bottom:6px;">🤖 LIVE PREDICTION TOOL</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:rgba(0,245,255,.6);
                margin-bottom:24px;">Enter patient metrics below to get an instant risk assessment</div>
    """, unsafe_allow_html=True)

    input_values = {}
    with st.form("predict_form"):
        st.caption(
            "")
        tab_clinical, tab_conditions, tab_genomic = st.tabs(
            ["🩺 Clinical & Demographics", "📋 Existing Conditions", "🧬 miRNA / Genomic Profile"])

        with tab_clinical:
            cols_layout = st.columns(3)
            for i, col in enumerate(clinical_cols):
                series = X_full[col]
                with cols_layout[i % 3]:
                    if col == "age":
                        input_values[col] = st.slider(pretty_label(col), int(series.min()), int(series.max()),
                                                      int(series.median()), key=col)
                    elif set(series.unique()).issubset({0, 1}):
                        label_1, label_0 = CLINICAL_BINARY_LABELS.get(
                            col, ("Yes", "No"))
                        val = st.selectbox(pretty_label(
                            col), [label_0, label_1], key=col)
                        input_values[col] = 1 if val == label_1 else 0
                    else:
                        input_values[col] = st.slider(pretty_label(col), float(series.min()), float(series.max()),
                                                      float(series.median()), key=col)

        with tab_conditions:
            for group_label, cols in onehot_groups.items():
                if group_label != "Existing Condition":
                    continue
                options = {c: c.split(
                    "existing_conditions_", 1)[-1] for c in cols}
                choice = st.radio(group_label, list(
                    options.values()), key=f"grp_{group_label}")
                for c, disp in options.items():
                    input_values[c] = 1 if disp == choice else 0

        with tab_genomic:
            prefix_map = {"miRNA Identifier": "mature_mirna_id_",
                          "Target Gene Symbol": "target_symbol_"}
            gcols = st.columns(2)
            gi = 0
            for group_label, cols in onehot_groups.items():
                if group_label == "Existing Condition":
                    continue
                prefix = prefix_map.get(group_label, "")
                options = {c: c.split(prefix, 1)
                           [-1].replace("_", " ") for c in cols}
                with gcols[gi % 2]:
                    choice = st.selectbox(group_label, list(
                        options.values()), key=f"grp_{group_label}")
                for c, disp in options.items():
                    input_values[c] = 1 if disp == choice else 0
                gi += 1

            st.markdown("**Target-prediction algorithm scores**")
            score_layout = st.columns(2)
            for i, col in enumerate(score_cols):
                series = X_full[col]
                with score_layout[i % 2]:
                    input_values[col] = st.slider(pretty_label(col), float(series.min()), float(series.max()),
                                                  float(series.median()), step=0.001, format="%.3f", key=col)

        submitted = st.form_submit_button(
            "🔍  RUN PREDICTION", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([input_values])[feature_cols]
        prediction = int(model.predict(input_df)[0])
        probability = model.predict_proba(input_df)[0]

        # The notebook is a binary 0/1 classifier; use the class labels
        # instead of assuming a particular column position.
        class_probs = dict(zip(model.classes_, probability))
        pos_prob = float(class_probs.get(1, 0.0)) * 100
        neg_prob = float(class_probs.get(0, 0.0)) * 100

        if prediction == 1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(255,58,92,.15),rgba(4,8,16,.95));
                        border:1px solid rgba(255,58,92,.5);border-radius:14px;padding:28px;
                        text-align:center;margin-top:16px;">
              <div style="font-size:2.5rem;margin-bottom:8px;">⚠️</div>
              <div style="font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:700;color:{RED};
                          text-shadow:0 0 24px {RED}88;margin-bottom:10px;">GASTRIC CANCER: POSITIVE</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:.9rem;color:rgba(255,200,200,.85);">
                Model Confidence: <b style="color:{RED};">{pos_prob:.1f}%</b> &nbsp;|&nbsp;
                Negative Probability: {neg_prob:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(0,255,153,.1),rgba(4,8,16,.95));
                        border:1px solid rgba(0,255,153,.45);border-radius:14px;padding:28px;
                        text-align:center;margin-top:16px;">
              <div style="font-size:2.5rem;margin-bottom:8px;">✅</div>
              <div style="font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:700;color:{GREEN};
                          text-shadow:0 0 24px {GREEN}88;margin-bottom:10px;">GASTRIC CANCER: NEGATIVE</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:.9rem;color:rgba(180,255,210,.85);">
                Model Confidence: <b style="color:{GREEN};">{neg_prob:.1f}%</b> &nbsp;|&nbsp;
                Positive Probability: {pos_prob:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pos_prob,
            title={"text": "Cancer Risk Probability (%)", "font": {
                "family": "Orbitron", "size": 13, "color": "#e8f4ff"}},
            number={"font": {"family": "Orbitron", "size": 28,
                             "color": RED if prediction == 1 else GREEN}, "suffix": "%"},
            gauge={"axis": {"range": [0, 100], "tickcolor": "#9abcd6", "tickfont": {"family": "IBM Plex Mono", "size": 9}},
                   "bar": {"color": RED if prediction == 1 else GREEN, "thickness": .25},
                   "bgcolor": "rgba(5,12,22,.8)", "bordercolor": "rgba(0,245,255,.2)",
                   "steps": [{"range": [0, 30], "color": "rgba(0,255,153,.08)"},
                             {"range": [30, 60],
                                 "color": "rgba(255,179,71,.08)"},
                             {"range": [60, 100], "color": "rgba(255,58,92,.08)"}],
                   "threshold": {"line": {"color": ORANGE, "width": 2}, "thickness": .75, "value": 50}},
        ))
        fig = apply_theme(fig, "", 280)
        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1 or pos_prob >= 50:
            st.error(
                "")

    section_divider()
    info_box(f"""<b>Model Summary:</b><br>
    Algorithm: Decision Tree (class_weight='balanced', max_depth={MODEL_MAX_DEPTH}, min_samples_leaf={MODEL_MIN_SAMPLES_LEAF})<br>
    Features Used: {len(feature_cols)}<br>
    Train/Test Split: {int((1-MODEL_TEST_SIZE)*100)}% / {int(MODEL_TEST_SIZE*100)}% (random_state=42, stratified)<br>
    Accuracy: {acc*100:.2f}% &nbsp;|&nbsp; Balanced Accuracy: {bal_acc*100:.2f}% &nbsp;|&nbsp;
    Recall (cancer): {rec*100:.2f}% &nbsp;|&nbsp; Precision (cancer): {prec*100:.2f}% &nbsp;|&nbsp;
    ROC-AUC: {auc:.4f}<br><br>
    <b>Honest takeaway:</b> ROC-AUC near 0.50 and balanced accuracy near 50% indicate the model has close to
    no real discriminative power on held-out data — the high recall comes at the cost of very low precision
    (lots of false positives), not from genuine signal in the features. This is a research/academic exercise,
    not a certified diagnostic device.""", "warn" if auc < 0.55 else "success")
