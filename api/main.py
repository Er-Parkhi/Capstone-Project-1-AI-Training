
from contextlib import asynccontextmanager
 
import pandas as pd

from fastapi import FastAPI, HTTPException
 
from api.model_loader import load_model

from api.schemas import TransactionInput, FraudPrediction
 
 
@asynccontextmanager

async def lifespan(app: FastAPI):

    app.state.model = load_model()

    yield
 
 
app = FastAPI(

    title="Fraud Transaction Detection API",

    description=(

        "Predicts potential fraudulent transactions using a trained "

        "Random Forest pipeline."

    ),

    version="1.0.0",

    lifespan=lifespan,

)
 
 
@app.get("/")

def home():

    return {

        "message": "Fraud Transaction Detection API",

        "docs": "/docs",

        "health": "/health",

    }
 
 
@app.get("/health")

def health():

    return {

        "status": "healthy",

        "model_loaded": hasattr(app.state, "model"),

    }
 
 
@app.post("/predict", response_model=FraudPrediction)

def predict(transaction: TransactionInput):

    model = getattr(app.state, "model", None)
 
    if model is None:

        raise HTTPException(

            status_code=503,

            detail="Model is not loaded.",

        )
 
    features = pd.DataFrame(

        [transaction.model_dump()]

    )
 
    try:

        probability = float(model.predict_proba(features)[0, 1])

        prediction = bool(probability >= 0.5)
 
        return FraudPrediction(

            is_fraud=prediction,

            fraud_probability=probability,

            model_name="RandomForest",

        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=f"Prediction failed: {type(exc).__name__}",

        ) from exc

