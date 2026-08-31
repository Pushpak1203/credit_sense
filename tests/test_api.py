from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "monthly_income": 55000.0,
    "repayment_history_score": 0.82,
    "utility_bill_regularity": 0.91,
    "years_in_business": 6,
    "debt_to_income_ratio": 0.28,
    "group_membership": 1,
    "income_stability_score": 0.88,
}


def test_score_valid_payload():
    response = client.post("/api/v1/score", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["score"] <= 100
    assert body["applicant_id"]


def test_score_invalid_payload():
    invalid = VALID_PAYLOAD.copy()
    invalid.pop("monthly_income")
    response = client.post("/api/v1/score", json=invalid)
    assert response.status_code == 422


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Credit Sense API is running"}
