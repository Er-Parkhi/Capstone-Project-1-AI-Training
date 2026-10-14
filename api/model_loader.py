
import joblib
 
from src.config import MODEL_DIR
 
MODEL_PATH = MODEL_DIR / "fraud_detection_pipeline.pkl"
 
 
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Trained model not found: {MODEL_PATH}. "

            "Run python -m src.train first."

        )
 
    return joblib.load(MODEL_PATH)

