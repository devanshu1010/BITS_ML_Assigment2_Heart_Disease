import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)


def generate_dataset(n=1000, seed=42):
    np.random.seed(seed)

    age      = np.random.randint(29, 77, n)
    sex      = np.random.randint(0, 2, n)
    cp       = np.random.randint(0, 4, n)
    trestbps = np.random.randint(94, 200, n)
    chol     = np.random.randint(126, 564, n)
    fbs      = np.random.randint(0, 2, n)
    restecg  = np.random.randint(0, 3, n)
    thalach  = np.random.randint(71, 202, n)
    exang    = np.random.randint(0, 2, n)
    oldpeak  = np.round(np.random.uniform(0, 6.2, n), 1)
    slope    = np.random.randint(0, 3, n)
    ca       = np.random.randint(0, 4, n)
    thal     = np.random.randint(0, 4, n)
    bmi      = np.round(np.random.uniform(18, 40, n), 1)

    risk_score = (
        0.3 * (age > 55).astype(int) +
        0.2 * sex +
        0.2 * (cp == 0).astype(int) +
        0.1 * (trestbps > 140).astype(int) +
        0.1 * (chol > 240).astype(int) +
        0.1 * exang +
        0.2 * (thalach < 130).astype(int) +
        0.1 * (ca > 1).astype(int) +
        np.random.uniform(0, 0.3, n)
    )
    target = (risk_score > 0.65).astype(int)

    return pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal, 'bmi': bmi, 'target': target
    })


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        'Accuracy':  round(accuracy_score(y, y_pred), 4),
        'AUC':       round(roc_auc_score(y, y_prob), 4),
        'Precision': round(precision_score(y, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y, y_pred, zero_division=0), 4),
        'F1 Score':  round(f1_score(y, y_pred, zero_division=0), 4),
        'MCC':       round(matthews_corrcoef(y, y_pred), 4),
    }, y_pred


def plot_comparison(results_df):
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    colors  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('ML Model Comparison — Heart Disease Classification',
                 fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        bars = ax.bar(results_df['Model'], results_df[metric],
                      color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'model_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(trained_models, X_test_scaled, y_test):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrices.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}\n")

    df.to_csv(os.path.join(BASE_DIR, 'heart_disease_data.csv'), index=False)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    test_data = X_test.copy()
    test_data['target'] = y_test.values
    test_data.to_csv(os.path.join(BASE_DIR, 'test_data.csv'), index=False)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows saved to test_data.csv\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes':         GaussianNB(),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
    }

    results = []
    trained_models = {}

    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        metrics, _ = evaluate(clf, X_test_scaled, y_test)

        fname = name.lower().replace(' ', '_')
        joblib.dump(clf, os.path.join(MODEL_DIR, f'{fname}.joblib'))
        trained_models[name] = clf

        print(f"{name:<25} Acc={metrics['Accuracy']}  AUC={metrics['AUC']}  F1={metrics['F1 Score']}")
        results.append({'Model': name, **metrics})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_DIR, 'results.csv'), index=False)

    print("\n--- Results Table ---")
    print(results_df.to_string(index=False))

    plot_comparison(results_df)
    plot_confusion_matrices(trained_models, X_test_scaled, y_test)

    print("\nAll models trained and saved to model/")
    print(f"Test data: test_data.csv ({len(X_test)} rows, {X_test.shape[1]} features + target)")


if __name__ == '__main__':
    main()
