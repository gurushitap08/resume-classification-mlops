import pandas as pd
import numpy as np
import re
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv("data/resume_data.csv")
print(f"Dataset shape: {df.shape}")

# ── 2. Clean Text ─────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'http\S+', ' ', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_resume'] = df['Resume_str'].apply(clean_text)

# ── 3. Encode Labels ──────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])

X = df['clean_resume']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. TF-IDF Vectorizer ──────────────────────────────────────
tfidf = TfidfVectorizer(max_features=1500, stop_words='english')

os.makedirs("models", exist_ok=True)

mlflow.set_experiment("Resume_Classification")

# ── 5. Model 1: PCA + Random Forest ──────────────────────────
print("\nTraining PCA + Random Forest...")
with mlflow.start_run(run_name="PCA_RandomForest"):
    pipeline_rf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1500, stop_words='english')),
        ('pca', PCA(n_components=100)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline_rf.fit(X_train, y_train)
    y_pred_rf = pipeline_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("pca_components", 100)
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.sklearn.log_model(pipeline_rf, "rf_model")
    joblib.dump(pipeline_rf, "models/rf_model.pkl")

# ── 6. Model 2: PCA + SVM ────────────────────────────────────
print("\nTraining PCA + SVM...")
with mlflow.start_run(run_name="PCA_SVM"):
    pipeline_svm = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1500, stop_words='english')),
        ('pca', PCA(n_components=100)),
        ('clf', SVC(kernel='rbf', C=1.0, random_state=42))
    ])
    pipeline_svm.fit(X_train, y_train)
    y_pred_svm = pipeline_svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {acc_svm:.4f}")
    mlflow.log_param("model", "SVM")
    mlflow.log_param("kernel", "rbf")
    mlflow.log_param("pca_components", 100)
    mlflow.log_metric("accuracy", acc_svm)
    mlflow.sklearn.log_model(pipeline_svm, "svm_model")
    joblib.dump(pipeline_svm, "models/svm_model.pkl")

# ── 7. Save Label Encoder ─────────────────────────────────────
joblib.dump(le, "models/label_encoder.pkl")

print("\n✅ Training complete!")
print(f"Random Forest Accuracy : {acc_rf:.4f}")
print(f"SVM Accuracy           : {acc_svm:.4f}")
print("\nBest model:", "Random Forest" if acc_rf > acc_svm else "SVM")