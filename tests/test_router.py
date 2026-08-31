from router.decision_router import route_decision


def test_auto_approve():
    assert route_decision(80, 0.90, "Low") == "auto_approve"


def test_auto_reject():
    assert route_decision(20, 0.90, "High") == "auto_reject"


def test_manual_review():
    assert route_decision(70, 0.60, "Low") == "manual_review"
