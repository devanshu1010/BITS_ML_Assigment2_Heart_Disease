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
    page_title="Heart Disease Classifier | ML Assignment 2",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e53935, #1565c0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .app-subtitle {
        text-align: center;
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .winner-box {
        background: linear-gradient(135deg, #2e7d32, #43a047);
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 12px 0;
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
    'Logistic Regression': 'Strong, interpretable baseline. Assumes a linear decision boundary between classes. Performs well when features have roughly linear relationships with the target. Good balance of precision and recall.',
    'Decision Tree':       'Highest precision among all models, but slightly lower recall. The tree depth is limited to 5 to avoid memorising training data. Human-readable decision rules are a major advantage.',
    'KNN':                 'Makes predictions by looking at the 7 nearest neighbours in feature space. No explicit training phase — it stores the training data. Slightly sensitive to irrelevant features but performs reliably here.',
    'Naive Bayes':         'Best overall performer: highest AUC, F1, and Recall. The Gaussian independence assumption holds reasonably well for this dataset. Extremely fast and well-suited to medical tasks where minimising false negatives matters.',
    'SVM':                 'Uses an RBF kernel to find a non-linear boundary. Competitive AUC and benefits greatly from the StandardScaler we applied. Training is slower than others but results are solid.',
}


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

st.markdown('<div class="app-title">Heart Disease Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">ML Assignment 2 &nbsp;·&nbsp; BITS Pilani WILP &nbsp;·&nbsp; M.Tech AIML &nbsp;·&nbsp; AIMLZG565</div>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("## Controls")
    st.markdown("### Upload Test CSV")
    uploaded_file = st.file_uploader(
        "Upload a CSV with all 14 features and the 'target' column",
        type=['csv']
    )
    st.markdown("### Select Classifier")
    chosen_model = st.selectbox("Model", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.markdown("**Dataset:** Heart Disease (synthetic)  \n**Instances:** 1 000  \n**Features:** 14  \n**Task:** Binary classification  \n`0` = No Disease | `1` = Disease")
    st.markdown("---")
    st.caption("Devanshu Chhipani · 2026AJ05742")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as err:
        st.error(f"Could not read file: {err}")
        st.stop()
else:
    default_csv = os.path.join(BASE_DIR, 'test_data.csv')
    if os.path.exists(default_csv):
        df = pd.read_csv(default_csv)
        st.info(f"No file uploaded — using the bundled **test_data.csv** ({len(df)} rows). Upload your own CSV from the sidebar.")
    else:
        st.warning("No test data found. Please upload a CSV file from the sidebar.")
        st.stop()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing these columns: {missing}")
    st.stop()
if 'target' not in df.columns:
    st.error("Your CSV must include a 'target' column (0 or 1).")
    st.stop()

if scaler is None:
    st.error("Scaler file not found in model/. Run train_models.py first.")
    st.stop()

X = df[FEATURES]
y = df['target']
X_scaled = scaler.transform(X)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Single Model",
    "🏆 All Models",
    "📈 Charts",
    "🗂 Data Preview",
])

with tab1:
    st.subheader(f"{chosen_model} — Evaluation")

    if chosen_model not in models:
        st.error(f"Model file for '{chosen_model}' was not found. Run train_models.py.")
    else:
        metrics, y_pred, y_prob = compute_metrics(models[chosen_model], X_scaled, y)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy",  metrics['Accuracy'])
        c2.metric("AUC",       metrics['AUC'])
        c3.metric("Precision", metrics['Precision'])
        c4.metric("Recall",    metrics['Recall'])
        c5.metric("F1 Score",  metrics['F1 Score'])
        c6.metric("MCC",       metrics['MCC'])

        st.markdown("---")
        left, right = st.columns(2)

        with left:
            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d',
                        cmap='Blues', ax=ax,
                        xticklabels=['No Disease', 'Disease'],
                        yticklabels=['No Disease', 'Disease'],
                        linewidths=1, linecolor='white')
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
            ax.set_title(chosen_model, fontweight='bold')
            st.pyplot(fig)
            plt.close()

        with right:
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y, y_prob)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(fpr, tpr, color='#1565c0', lw=2.5,
                     label=f"AUC = {metrics['AUC']}")
            ax2.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random')
            ax2.fill_between(fpr, tpr, alpha=0.08, color='#1565c0')
            ax2.set_xlabel('False Positive Rate', fontweight='bold')
            ax2.set_ylabel('True Positive Rate', fontweight='bold')
            ax2.set_title(f'ROC — {chosen_model}', fontweight='bold')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.25)
            st.pyplot(fig2)
            plt.close()

        st.markdown("#### Classification Report")
        report = classification_report(y, y_pred,
                                       target_names=['No Disease', 'Disease'],
                                       output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with tab2:
    st.subheader("All Models — Side-by-Side Comparison")

    rows = []
    for name in MODEL_FILES:
        if name not in models:
            continue
        m_metrics, _, _ = compute_metrics(models[name], X_scaled, y)
        rows.append({'Model': name, **m_metrics})

    comparison = pd.DataFrame(rows)

    def highlight_best(col):
        return ['background-color: #c8e6c9; font-weight: bold'
                if v == col.max() else '' for v in col]

    metric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    st.dataframe(
        comparison.style.apply(highlight_best, subset=metric_cols),
        use_container_width=True,
        height=215
    )

    best_idx  = comparison['F1 Score'].idxmax()
    best_name = comparison.loc[best_idx, 'Model']
    best_f1   = comparison.loc[best_idx, 'F1 Score']
    best_acc  = comparison.loc[best_idx, 'Accuracy']

    st.markdown(f"""
    <div class="winner-box">
        🥇 Best Model: {best_name} &nbsp;|&nbsp; F1 = {best_f1} &nbsp;|&nbsp; Accuracy = {best_acc}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Model Observations")

    obs_rows = [{'Model': k, 'Observation': v} for k, v in MODEL_OBSERVATIONS.items()]
    obs_df   = pd.DataFrame(obs_rows)
    st.dataframe(obs_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"**Overall Winner → {best_name}**  \n"
        f"It achieves the best F1 Score ({best_f1}) and the highest AUC on this dataset. "
        f"In a heart disease context, high recall is critical — missing a positive case is far more "
        f"dangerous than a false alarm — and {best_name} leads on that metric too."
    )

with tab3:
    st.subheader("Visual Comparison")

    plot_rows = []
    for name in MODEL_FILES:
        if name not in models:
            continue
        m_metrics, _, _ = compute_metrics(models[name], X_scaled, y)
        plot_rows.append({'Model': name, **m_metrics})
    plot_df = pd.DataFrame(plot_rows)
    short_names = ['LR', 'DT', 'KNN', 'NB', 'SVM']

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Metric Bar Chart")
        bar_metrics = ['Accuracy', 'AUC', 'F1 Score', 'MCC']
        bar_colors  = ['#1565c0', '#e53935', '#43a047', '#ff8f00']
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(plot_df))
        w = 0.2
        for i, (m_name, color) in enumerate(zip(bar_metrics, bar_colors)):
            ax.bar(x + i * w, plot_df[m_name], w,
                   label=m_name, color=color, alpha=0.85, edgecolor='white')
        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels(short_names)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with right_col:
        st.markdown("#### ROC Curves — All Models")
        palette = ['#1565c0', '#e53935', '#43a047', '#ff8f00', '#7b1fa2']
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
        for name, color in zip(MODEL_FILES, palette):
            if name not in models:
                continue
            _, _, yb = compute_metrics(models[name], X_scaled, y)
            fpr, tpr, _ = roc_curve(y, yb)
            auc_val = roc_auc_score(y, yb)
            ax2.plot(fpr, tpr, lw=2, color=color,
                     label=f'{name} (AUC={auc_val:.3f})')
        ax2.set_xlabel('False Positive Rate', fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontweight='bold')
        ax2.set_title('ROC Curves', fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.25)
        st.pyplot(fig2)
        plt.close()

    st.markdown("#### Confusion Matrices — All Models")
    fig3, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, name in zip(axes, MODEL_FILES):
        if name not in models:
            continue
        _, yp, _ = compute_metrics(models[name], X_scaled, y)
        cm = confusion_matrix(y, yp)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                    linewidths=0.5)
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with tab4:
    st.subheader("Test Data Preview")
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

    left_d, right_d = st.columns(2)
    with left_d:
        st.markdown("**First 20 rows**")
        st.dataframe(df.head(20), use_container_width=True)
    with right_d:
        st.markdown("**Summary Statistics**")
        st.dataframe(df.describe().round(3), use_container_width=True)

    col_pie, col_heat = st.columns([1, 2])
    with col_pie:
        st.markdown("**Target Distribution**")
        counts = df['target'].value_counts()
        fig_p, ax_p = plt.subplots(figsize=(4, 4))
        ax_p.pie(counts, labels=['No Disease', 'Disease'],
                 autopct='%1.1f%%', colors=['#43a047', '#e53935'],
                 startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax_p.set_title('Class Distribution')
        st.pyplot(fig_p)
        plt.close()

    with col_heat:
        st.markdown("**Feature Correlation**")
        fig_h, ax_h = plt.subplots(figsize=(10, 7))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f',
                    cmap='coolwarm', ax=ax_h, linewidths=0.4,
                    annot_kws={'size': 7})
        ax_h.set_title('Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close()

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:0.82rem;'>"
    "ML Assignment 2 · BITS Pilani WILP · M.Tech AIML · AIMLZG565<br>"
    "Devanshu Chhipani · 2026AJ05742 · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
