from model.predict import predict

SAMPLE_APPLICANT = {
    "monthly_income": 55000.0,
    "repayment_history_score": 0.82,
    "utility_bill_regularity": 0.91,
    "years_in_business": 6,
    "debt_to_income_ratio": 0.28,
    "group_membership": 1,
    "income_stability_score": 0.88,
}


def test_predict_returns_expected_keys():
    result = predict(SAMPLE_APPLICANT)
    expected = {
        "score",
        "probability",
        "risk_band",
        "top_factors",
        "recommended_action",
        "confidence",
    }
    assert expected.issubset(result.keys())
    assert len(result["top_factors"]) == 3


def test_score_range():
    result = predict(SAMPLE_APPLICANT)
    assert 0 <= result["score"] <= 100


def test_risk_band():
    result = predict(SAMPLE_APPLICANT)
    assert result["risk_band"] in {"Low", "Medium", "High"}
