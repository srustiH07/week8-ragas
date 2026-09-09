from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="W8D1 ML API",
    description="Dockerised Machine Learning API for MLOps demonstration",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "W8D1 ML API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ml-api"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        return {
            "prediction": "unknown",
            "confidence": 0.0
        }

    positive_words = [
        "good", "great", "excellent", "happy",
        "success", "successful", "love", "best"
    ]

    negative_words = [
        "bad", "poor", "terrible", "sad",
        "failure", "failed", "hate", "worst"
    ]

    words = text.lower().split()

    positive_score = sum(word.strip(".,!?") in positive_words for word in words)
    negative_score = sum(word.strip(".,!?") in negative_words for word in words)

    if positive_score > negative_score:
        prediction = "positive"
    elif negative_score > positive_score:
        prediction = "negative"
    else:
        prediction = "neutral"

    total = positive_score + negative_score

    confidence = (
        round(max(positive_score, negative_score) / total, 2)
        if total > 0
        else 0.50
    )

    return {
        "text": text,
        "prediction": prediction,
        "confidence": confidence
    }