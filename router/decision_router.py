from config.config import (
    AUTO_APPROVE_SCORE_THRESHOLD,
    AUTO_REJECT_SCORE_THRESHOLD,
    CONFIDENCE_THRESHOLD,
)


def route_decision(score: int, confidence: float, risk_band: str) -> str:
    if score >= AUTO_APPROVE_SCORE_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD:
        return "auto_approve"
    if score < AUTO_REJECT_SCORE_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD:
        return "auto_reject"
    return "manual_review"
