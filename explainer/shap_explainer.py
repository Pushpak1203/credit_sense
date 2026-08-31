from __future__ import annotations

import numpy as np

TEMPLATES = {
    "repayment_history_score": {
        "positive": "Strong track record of repaying past dues on time",
        "negative": "Inconsistent or missed repayments in the past",
    },
    "utility_bill_regularity": {
        "positive": "Utility bills paid consistently every month",
        "negative": "Frequent gaps or delays in utility bill payments",
    },
    "debt_to_income_ratio": {
        "positive": "Debt level is manageable relative to income",
        "negative": "High debt relative to income raises repayment risk",
    },
    "monthly_income": {
        "positive": "Income level is sufficient to support loan repayment",
        "negative": "Low income relative to requested loan amount",
    },
    "group_membership": {
        "positive": "SHG/co-op membership shows community-backed repayment behaviour",
        "negative": "No group or co-op affiliation on record",
    },
    "years_in_business": {
        "positive": "Established business history increases trust",
        "negative": "Limited business history increases uncertainty",
    },
    "income_stability_score": {
        "positive": "Income is stable and predictable month-to-month",
        "negative": "Income shows significant month-to-month variation",
    },
}


def _normalize_shap_values(shap_values) -> np.ndarray:
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[0]
        if values.shape[1] > 1:
            values = values[:, 1]
        else:
            values = values[:, 0]
    elif values.ndim == 2:
        values = values[0]
    return values.astype(float)


def explain(shap_values, feature_names, feature_values) -> list[dict]:
    values = _normalize_shap_values(shap_values)
    if len(values) != len(feature_names):
        raise ValueError("SHAP values and feature names must have the same length")

    ranked = sorted(
        zip(feature_names, values),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:3]

    factors = []
    for feature, shap_value in ranked:
        impact = "positive" if float(shap_value) >= 0 else "negative"
        plain_english = TEMPLATES.get(
            feature,
            {
                "positive": f"{feature} improves the estimated repayment profile",
                "negative": f"{feature} weakens the estimated repayment profile",
            },
        )[impact]
        factors.append(
            {
                "feature": feature,
                "impact": impact,
                "plain_english": plain_english,
            }
        )
    return factors
