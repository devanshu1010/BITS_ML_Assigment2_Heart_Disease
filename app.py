import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import warnings

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Heart Disease Classification | BITS Pilani ML",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Theme-Adaptive CSS (supports both Light & Dark modes naturally)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .header-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 0.5rem;
    }

    .winner-card {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 8px;
        padding: 14px 18px;
        margin: 16px 0;
        color: var(--text-color, inherit);
    }

    .winner-card-title {
        font-size: 1rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        color: #10b981;
    }

    .winner-card-desc {
        font-size: 0.9rem;
        color: var(--text-color, inherit);
        opacity: 0.9;
        line-height: 1.5;
    }

    .section-label {
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-color, inherit);
        opacity: 0.75;
        margin-bottom: 8px;
    }

    /* Subtle card border for native st.metric cards in both light and dark modes */
    [data-testid="stMetric"] {
        background-color: rgba(148, 163, 184, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
    }

    /* Style matplotlib figure cards cleanly */
    .stPlot {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree':       'decision_tree.joblib',
    'KNN':                 'knn.joblib',
    'Naive Bayes':         'naive_bayes.joblib',
    'SVM':                 'svm.joblib',
}

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'bmi']

MODEL_OBSERVATIONS = {
    'Logistic Regression': 'Strong, interpretable linear baseline. Weights correspond directly to log-odds of heart disease risk factors. Balanced precision and recall with robust generalization.',
    'Decision Tree':       'Achieves highest precision (0.8899). Depth constrained to 5 layers to control variance and prevent overfitting. Captures orthogonal decision thresholds naturally.',
    'KNN':                 'Evaluates similarity using Euclidean distance over 7 standardized nearest neighbors. Reliable non-parametric baseline, though computationally heavier during larger inference passes.',
    'Naive Bayes':         'Best overall classifier on this dataset (AUC 0.9202, F1 0.8712). Class-conditional Gaussian independence performs exceptionally well, minimizing false negatives (Recall 0.9200) — critical in medical screening.',
    'SVM':                 'RBF kernel maps clinical attributes into higher-dimensional margin space. Solid AUC of 0.9004; benefits directly from the feature-wise StandardScaler transformation.',
}

GITHUB_REPO_URL = "https://github.com/devanshu1010/BITS_ML_Assigment2_Heart_Disease"


@st.cache_resource
def load_artifacts():
    loaded = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            loaded[name] = joblib.load(path)

    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return loaded, scaler


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        'Accuracy':  round(accuracy_score(y, y_pred), 4),
        'AUC':       round(roc_auc_score(y, y_prob), 4),
        'Precision': round(precision_score(y, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y, y_pred, zero_division=0), 4),
        'F1 Score':  round(f1_score(y, y_pred, zero_division=0), 4),
        'MCC':       round(matthews_corrcoef(y, y_pred), 4),
    }, y_pred, y_prob


models, scaler = load_artifacts()

# App Header (Theme-Adaptive)
st.markdown('<div class="header-badge">BITS Pilani · WILP M.Tech (AIML) · AIMLZG565</div>', unsafe_allow_html=True)
head_col, gh_col = st.columns([3.5, 1])
with head_col:
    st.title("Heart Disease Classification & Model Benchmarking")
    st.caption("Assignment 2 — End-to-end multi-classifier implementation, evaluation, and deployment")
with gh_col:
    st.write("")
    st.link_button("⭐ GitHub Repository", GITHUB_REPO_URL, help="View source code on GitHub")

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown('<div class="section-label">Upload Test Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload test CSV (features + target)",
        type=['csv'],
        help="Upload CSV containing 14 features and binary target column"
    )

    st.markdown('<div class="section-label">Active Model Selection</div>', unsafe_allow_html=True)
    chosen_model = st.selectbox("Classifier", list(MODEL_FILES.keys()))

    st.markdown("---")
    st.markdown('<div class="section-label">Dataset Specifications</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Cohort Size:** 1,000 instances
    - **Feature Dimensions:** 14 clinical attributes
    - **Target Column:** `target` (0 = No Disease, 1 = Disease)
    - **Input Requirements:** ≥ 12 features, ≥ 500 instances
    """)

    st.markdown("---")
    st.link_button("🔗 Source Repository", GITHUB_REPO_URL)
    st.caption("Student: Devanshu Chhipani · ID: 2026AJ05742")

# Data Ingestion
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded **{uploaded_file.name}** ({df.shape[0]} rows × {df.shape[1]} columns)")
    except Exception as err:
        st.error(f"Error parsing CSV file: {err}")
        st.stop()
else:
    default_csv = os.path.join(BASE_DIR, 'test_data.csv')
    if os.path.exists(default_csv):
        df = pd.read_csv(default_csv)
        st.info(f"Using default verification split: **test_data.csv** ({len(df)} instances). You can upload a custom test split from the sidebar.")
    else:
        st.warning("No test data found. Please upload a CSV file via the sidebar.")
        st.stop()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()
if 'target' not in df.columns:
    st.error("Missing required binary outcome column: 'target'")
    st.stop()

if scaler is None:
    st.error("StandardScaler artifact not found in model/. Run train_models.py to generate artifacts.")
    st.stop()

X = df[FEATURES]
y = df['target']
X_scaled = scaler.transform(X)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Model Performance",
    "Comprehensive Comparison",
    "Diagnostic Visualizations",
    "Data Exploration"
])

# Tab 1: Single Model Evaluation
with tab1:
    st.subheader(f"Evaluation Overview — {chosen_model}")

    if chosen_model not in models:
        st.error(f"Model file for '{chosen_model}' was not found. Please train models.")
    else:
        metrics, y_pred, y_prob = compute_metrics(models[chosen_model], X_scaled, y)

        # Native theme-adaptive metric components
        cols = st.columns(6)
        cols[0].metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        cols[1].metric("AUC Score", f"{metrics['AUC']:.4f}")
        cols[2].metric("Precision", f"{metrics['Precision']:.4f}")
        cols[3].metric("Recall", f"{metrics['Recall']:.4f}")
        cols[4].metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        cols[5].metric("MCC Score", f"{metrics['MCC']:.4f}")

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(4.8, 3.8))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease (0)', 'Disease (1)'],
                yticklabels=['No Disease (0)', 'Disease (1)'],
                cbar=False, linewidths=1, linecolor='#e2e8f0',
                annot_kws={'size': 13, 'weight': 'bold'}
            )
            ax.set_xlabel('Predicted Outcome', fontsize=9.5, fontweight='bold', color='#1e293b')
            ax.set_ylabel('True Outcome', fontsize=9.5, fontweight='bold', color='#1e293b')
            ax.tick_params(labelsize=8.5, colors='#334155')
            ax.set_title(f"{chosen_model}", fontsize=11, fontweight='bold', color='#0f172a', pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_roc:
            st.markdown('<div class="section-label">ROC Curve Analysis</div>', unsafe_allow_html=True)
            fpr, tpr, _ = roc_curve(y, y_prob)
            fig_roc, ax_roc = plt.subplots(figsize=(4.8, 3.8))
            fig_roc.patch.set_facecolor('#ffffff')
            ax_roc.set_facecolor('#ffffff')
            ax_roc.plot(fpr, tpr, color='#1e3a8a', lw=2.2, label=f"AUC = {metrics['AUC']:.4f}")
            ax_roc.plot([0, 1], [0, 1], linestyle='--', color='#94a3b8', lw=1.2, label='Chance')
            ax_roc.fill_between(fpr, tpr, alpha=0.08, color='#1e3a8a')
            ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=9.5, fontweight='bold', color='#1e293b')
            ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=9.5, fontweight='bold', color='#1e293b')
            ax_roc.tick_params(labelsize=8.5, colors='#334155')
            ax_roc.set_title(f"ROC — {chosen_model}", fontsize=11, fontweight='bold', color='#0f172a', pad=10)
            ax_roc.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=8.5, loc='lower right')
            ax_roc.grid(True, color='#f1f5f9', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig_roc)
            plt.close()

        st.markdown('<div class="section-label">Detailed Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y, y_pred, target_names=['No Disease', 'Disease'], output_dict=True)
        report_df = pd.DataFrame(report).T.round(4)
        st.dataframe(report_df)

# Tab 2: All Models Comparison
with tab2:
    st.subheader("Model Benchmarking Matrix")

    rows = []
    for name in MODEL_FILES:
        if name not in models:
            continue
        m_metrics, _, _ = compute_metrics(models[name], X_scaled, y)
        rows.append({'ML Model Name': name, **m_metrics})

    comparison = pd.DataFrame(rows)
    metric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']

    comparison_display = comparison.copy()
    for col in metric_cols:
        comparison_display[col] = comparison_display[col].apply(lambda v: f"{v:.4f}")

    st.dataframe(comparison_display, hide_index=True)

    best_idx  = comparison['F1 Score'].idxmax()
    best_name = comparison.loc[best_idx, 'ML Model Name']
    best_f1   = comparison.loc[best_idx, 'F1 Score']
    best_auc  = comparison.loc[best_idx, 'AUC']
    best_rec  = comparison.loc[best_idx, 'Recall']

    st.markdown(f"""
    <div class="winner-card">
        <div class="winner-card-title">
            <span>🏆</span> Benchmark Winner: {best_name}
        </div>
        <div class="winner-card-desc">
            <strong>{best_name}</strong> demonstrated superior diagnostic stability on this patient test set with an 
            <strong>F1 Score of {best_f1:.4f}</strong>, <strong>AUC of {best_auc:.4f}</strong>, and an industry-vital 
            <strong>Recall of {best_rec:.4f}</strong>. Minimizing false negatives (undetected disease) is the paramount clinical objective.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Architectural & Performance Observations")
    obs_df = pd.DataFrame([{'ML Model Name': k, 'Observation about model performance': v} for k, v in MODEL_OBSERVATIONS.items()])
    st.dataframe(obs_df, hide_index=True)

# Tab 3: Visual Comparison
with tab3:
    st.subheader("Comparative Diagnostic Analytics")

    plot_rows = []
    for name in MODEL_FILES:
        if name not in models:
            continue
        m_metrics, _, _ = compute_metrics(models[name], X_scaled, y)
        plot_rows.append({'Model': name, **m_metrics})
    plot_df = pd.DataFrame(plot_rows)
    short_names = ['LR', 'DT', 'KNN', 'NB', 'SVM']

    col_bar, col_multi_roc = st.columns(2)

    with col_bar:
        st.markdown('<div class="section-label">Multi-Metric Side-by-Side Comparison</div>', unsafe_allow_html=True)
        fig_bar, ax_bar = plt.subplots(figsize=(6.5, 4.2))
        fig_bar.patch.set_facecolor('#ffffff')
        ax_bar.set_facecolor('#ffffff')
        bar_metrics = ['Accuracy', 'AUC', 'F1 Score', 'MCC']
        bar_colors  = ['#1e3a8a', '#2563eb', '#0d9488', '#d97706']
        x = np.arange(len(plot_df))
        w = 0.18
        for i, (m_name, color) in enumerate(zip(bar_metrics, bar_colors)):
            ax_bar.bar(x + i * w, plot_df[m_name], w, label=m_name, color=color, edgecolor='#ffffff', linewidth=0.5)
        ax_bar.set_xticks(x + w * 1.5)
        ax_bar.set_xticklabels(short_names, fontweight='bold', color='#1e293b')
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_ylabel('Score Metric', fontsize=9, fontweight='bold', color='#1e293b')
        ax_bar.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=8, loc='upper right')
        ax_bar.grid(axis='y', color='#f1f5f9', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close()

    with col_multi_roc:
        st.markdown('<div class="section-label">Multi-Model ROC Trajectories</div>', unsafe_allow_html=True)
        fig_mroc, ax_mroc = plt.subplots(figsize=(6.5, 4.2))
        fig_mroc.patch.set_facecolor('#ffffff')
        ax_mroc.set_facecolor('#ffffff')
        colors_roc = ['#1e3a8a', '#2563eb', '#0d9488', '#d97706', '#e11d48']
        ax_mroc.plot([0, 1], [0, 1], linestyle='--', color='#94a3b8', lw=1.2, label='Chance')
        for (m_name, color) in zip(MODEL_FILES, colors_roc):
            if m_name not in models:
                continue
            _, _, yb = compute_metrics(models[m_name], X_scaled, y)
            fpr, tpr, _ = roc_curve(y, yb)
            score = roc_auc_score(y, yb)
            ax_mroc.plot(fpr, tpr, lw=1.8, color=color, label=f"{m_name} ({score:.3f})")
        ax_mroc.set_xlabel('False Positive Rate', fontsize=9, fontweight='bold', color='#1e293b')
        ax_mroc.set_ylabel('True Positive Rate', fontsize=9, fontweight='bold', color='#1e293b')
        ax_mroc.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=7.5, loc='lower right')
        ax_mroc.grid(True, color='#f1f5f9', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig_mroc)
        plt.close()

    st.markdown('<div class="section-label" style="margin-top: 14px;">Confusion Matrix Matrix (All Classifiers)</div>', unsafe_allow_html=True)
    fig_cm_all, axes_all = plt.subplots(1, 5, figsize=(18, 3.4))
    fig_cm_all.patch.set_facecolor('#ffffff')
    for ax, name in zip(axes_all, MODEL_FILES):
        ax.set_facecolor('#ffffff')
        if name not in models:
            continue
        _, yp, _ = compute_metrics(models[name], X_scaled, y)
        cm_sub = confusion_matrix(y, yp)
        sns.heatmap(
            cm_sub, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
            cbar=False, linewidths=0.8, linecolor='#ffffff',
            annot_kws={'size': 10, 'weight': 'bold'}
        )
        ax.set_title(name, fontsize=9.5, fontweight='bold', color='#0f172a')
        ax.set_xlabel('Predicted', fontsize=8, color='#334155')
        ax.set_ylabel('Actual', fontsize=8, color='#334155')
    plt.tight_layout()
    st.pyplot(fig_cm_all)
    plt.close()

# Tab 4: Data Exploration
with tab4:
    st.subheader("Data Distribution & Feature Profiling")

    top_l, top_r = st.columns([1, 2])

    with top_l:
        st.markdown('<div class="section-label">Cohort Prevalence (Target)</div>', unsafe_allow_html=True)
        counts = df['target'].value_counts()
        fig_donut, ax_donut = plt.subplots(figsize=(4, 3.4))
        fig_donut.patch.set_facecolor('#ffffff')
        ax_donut.set_facecolor('#ffffff')
        wedges, texts, autotexts = ax_donut.pie(
            counts, labels=['No Disease', 'Heart Disease'],
            autopct='%1.1f%%', startangle=90,
            colors=['#0d9488', '#1e3a8a'],
            textprops={'color': '#1e293b', 'fontsize': 8.5, 'weight': 'bold'},
            wedgeprops={'width': 0.52, 'edgecolor': '#ffffff', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_fontsize(8.5)
            at.set_weight('bold')
            at.set_color('#ffffff')
        plt.tight_layout()
        st.pyplot(fig_donut)
        plt.close()

    with top_r:
        st.markdown('<div class="section-label">Sample Data Records (Head 5)</div>', unsafe_allow_html=True)
        st.dataframe(df.head(5))

    st.markdown('<div class="section-label" style="margin-top: 10px;">Inter-Feature Correlation Matrix</div>', unsafe_allow_html=True)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 5.5))
    fig_corr.patch.set_facecolor('#ffffff')
    ax_corr.set_facecolor('#ffffff')
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr, annot=True, fmt='.2f',
        cmap='vlag', center=0,
        ax=ax_corr, linewidths=0.5, linecolor='#ffffff',
        annot_kws={'size': 7.5, 'color': '#0f172a'},
        cbar_kws={'shrink': 0.8}
    )
    ax_corr.tick_params(labelsize=8, colors='#334155')
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close()

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:8px;'>"
    "BITS Pilani WILP · M.Tech in Artificial Intelligence & Machine Learning · Machine Learning (AIMLZG565)"
    "</div>",
    unsafe_allow_html=True
)
