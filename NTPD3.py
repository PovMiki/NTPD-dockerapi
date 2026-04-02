from fastapi import FastAPI, HTTPException
from sklearn.linear_model import LinearRegression
import numpy as np
from pydantic import BaseModel
import redis

app = FastAPI()

X = np.array([[1], [2], [3], [5], [8], [10]])
y = np.array([15, 28, 40, 62, 85, 98])

model = LinearRegression()
model.fit(X, y)

class PredictionInput(BaseModel):
    hours: float

cache = redis.Redis(host='redis-db', port=6379, decode_responses=True)
@app.get("/redis-test")
def test_redis():
    try:
        cache.set("test_key", "polaczenie dziala")
        value = cache.get("test_key")
        return {"redis_status": "success", "data": value}
    except Exception as e:
        return {"redis_status": "error", "message": str(e)}

@app.post("/predict")
async def predict(data: PredictionInput):
    if data.hours < 0:
        raise HTTPException(
            status_code=400,
            detail="godzina nie moze byc ujemna"
        )
    if data.hours > 24:
        raise HTTPException(
            status_code=400,
            detail="godzina nie moze byc > 24"
        )
    prediction = model.predict([[data.hours]])
    score = round(float(prediction[0]), 2)

    return {
        "input_hours": data.hours,
        "predicted_score": score
    }

@app.get("/")
def read_root():
    return {"Hello": "Test123"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

@app.get("/info")
def get_info():
    return {
        "model_type": "LinearRegression",
        "training_samples": len(X),
        "features": ["study_hours"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "message": "jest git"
    }


