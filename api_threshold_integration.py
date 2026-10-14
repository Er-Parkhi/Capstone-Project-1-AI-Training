"""Example only: load a selected threshold without breaking old model artifacts."""
import json
from pathlib import Path

def load_fraud_threshold(path="models/fraud_threshold.json", default=0.5):
    p = Path(path)
    if not p.exists():
        return float(default)
    payload = json.loads(p.read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("Configured fraud threshold must be between 0 and 1.")
    return threshold

# In api/main.py, replace a hard-coded `probability >= 0.5` comparison with:
# from api_threshold_integration import load_fraud_threshold
# FRAUD_THRESHOLD = load_fraud_threshold()
# is_fraud = probability >= FRAUD_THRESHOLD
