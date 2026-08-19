# 🧬 Gastric Cancer Risk Prediction App

An interactive Streamlit application designed to evaluate patient gastric cancer risk using clinical metrics, diagnostic imaging indicators, and genomic biomarkers.

---

## 📌 Problem Statement & Core Challenges

Diagnosing gastric cancer early is critical, but machine learning on this dataset presents three key challenges:

1. **Severe Class Imbalance (~9:1 Ratio)**  
   The dataset contains ~90.1% negative and ~9.9% positive gastric cancer cases. Standard unweighted models (Logistic Regression, default XGBoost) default to predicting 0 for all patients, yielding **0% cancer recall**.

2. **Weak Linear Signals**  
   Individual clinical and miRNA features exhibit very low linear correlation ($|r| \le 0.04$) with cancer diagnosis. Linear classifiers fail to find meaningful linear separation.

3. **High Overfitting Risk**  
   Unpruned decision trees memorize noisy training patterns (100% train accuracy) but fail to generalize to unseen test patients.

---

## 💡 Model Approach & Performance

We implemented a **regularized, class-balanced Decision Tree** (`class_weight='balanced'`, `max_depth=8`, `min_samples_leaf=50`) optimized for high medical sensitivity:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Recall (Cancer)** | **67.9%** | High sensitivity focus to minimize missed diagnoses |
| **Accuracy** | **81.7%** | Overall classification accuracy |
| **Precision** | **9.9%** | Trade-off from class-balanced weighting |
| **Balanced Accuracy** | **50.1%** | Average recall across positive and negative classes |

---

## 🖥️ App Features

* **🤖 Live Risk Predictor**: Dynamic risk calculation with a real-time **Speedometer Gauge Chart** (Low, Moderate, High Risk tiers).
* **📁 Dataset Overview**: Class distribution breakdown and age distribution histogram.
* **🔬 Feature Analysis**: Chi-Square significance rankings ($p$-values) and Pearson correlation coefficients ($r$).

---

## 🚀 How to Run Locally

```bash
# Navigate to app directory
cd app/Gastric_Cancer_Prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run gastric_cancer_prediction.py
```
Open `http://localhost:8501` in your web browser.
