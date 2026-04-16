from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

app = FastAPI(title="Resume Classification API")

# Load model and label encoder
model = joblib.load("models/svm_model.pkl")
le = joblib.load("models/label_encoder.pkl")

def clean_text(text):
    text = re.sub(r'http\S+', ' ', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class ResumeInput(BaseModel):
    resume_text: str

@app.get("/")
def home():
    return {"message": "Resume Classification API is running!"}

@app.post("/predict")
def predict(data: ResumeInput):
    cleaned = clean_text(data.resume_text)
    prediction = model.predict([cleaned])
    category = le.inverse_transform(prediction)[0]
    return {
        "predicted_category": category
    }