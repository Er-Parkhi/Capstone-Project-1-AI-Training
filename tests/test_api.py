

import pytest

from fastapi.testclient import TestClient
 
from api.main import app
 
 
@pytest.fixture(scope="module")

def client():

    with TestClient(app) as test_client:

        yield test_client
 
 
def test_home_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["docs"] == "/docs"
 
 
def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"

    assert response.json()["model_loaded"] is True
 
 
def test_predict_valid_transaction(client):

    payload = {

        "step": 1,

        "type": "TRANSFER",

        "amount": 181.0,

        "oldbalanceOrg": 181.0,

        "newbalanceOrig": 0.0,

        "oldbalanceDest": 0.0,

        "newbalanceDest": 0.0,

    }
 
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
 
    body = response.json()

    assert isinstance(body["is_fraud"], bool)

    assert 0.0 <= body["fraud_probability"] <= 1.0

    assert body["model_name"] == "RandomForest"
 
 
def test_predict_rejects_missing_fields(client):

    response = client.post(

        "/predict",

        json={"step": 1, "type": "TRANSFER", "amount": 181.0},

    )

    assert response.status_code == 422
 
 
def test_predict_rejects_negative_amount(client):

    payload = {

        "step": 1,

        "type": "TRANSFER",

        "amount": -181.0,

        "oldbalanceOrg": 181.0,

        "newbalanceOrig": 0.0,

        "oldbalanceDest": 0.0,

        "newbalanceDest": 0.0,

    }
 
    response = client.post("/predict", json=payload)

    assert response.status_code == 422

 