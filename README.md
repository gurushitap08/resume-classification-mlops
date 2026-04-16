# Resume Classification Automation using MLOps

A complete MLOps pipeline for classifying resumes into job categories using Machine Learning.

## 📌 Project Overview
This project automates resume classification using multiple ML techniques and follows MLOps best practices including version control, containerization, and CI/CD pipelines.

## 📂 Dataset
- Source: Kaggle Resume Dataset
- Total Resumes: 2,484
- Categories: 24 job categories (HR, Engineering, Finance, etc.)

## 🤖 ML Models Used
| Model | Technique | Accuracy |
|-------|-----------|----------|
| Random Forest | PCA + TF-IDF + Random Forest | 60.36% |
| SVM | PCA + TF-IDF + SVM | 64.39% |

## 🛠️ Tech Stack
- **ML**: Scikit-learn, PCA, TF-IDF, Random Forest, SVM
- **API**: FastAPI + Uvicorn
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Data Versioning**: DVC
- **Experiment Tracking**: MLflow

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/gurushitap08/resume-classification-mlops.git
cd resume-classification-mlops
```

### 2. Create virtual environment
```bash
python -m venv resume-mlops
resume-mlops\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model
```bash
python src/train.py
```

### 4. Run FastAPI
```bash
uvicorn api.main:app --reload
```

### 5. Run with Docker
```bash
docker build -t resume-classification .
docker run -p 8000:8000 resume-classification
```

## 📊 API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Classify resume |

## 🔁 CI/CD Pipeline
Automatically triggers on every push to main branch:
- ✅ Install dependencies
- ✅ Syntax check
- ✅ Build Docker image

## 📁 Project Structure
```
resume-classification-mlops/
├── data/               # Dataset (tracked by DVC)
├── src/                # Training scripts
├── api/                # FastAPI app
├── models/             # Saved models
├── .github/workflows/  # CI/CD pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```