from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import FEATURE_NAMES_PATH, MODEL_PATH, SHAP_EXPLAINER_PATH
from explainer.shap_explainer import explain
from router.decision_router import route_decision


@lru_cache(maxsize=1)
def _load_artifacts():
    required_paths = [MODEL_PATH, SHAP_EXPLAINER_PATH, FEATURE_NAMES_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model artifacts are missing: "
            + ", ".join(missing)
            + ". Run: python data/generate_synthetic_data.py && python model/train.py"
        )

    model = joblib.load(MODEL_PATH)
    shap_explainer = joblib.load(SHAP_EXPLAINER_PATH)
    with FEATURE_NAMES_PATH.open("r", encoding="utf-8") as file:
        feature_names = json.load(file)
    return model, shap_explainer, feature_names


def _risk_band(score: int) -> str:
    if score >= 65:
        return "Low"
    if score >= 40:
        return "Medium"
    return "High"


def _confidence(probability: float) -> float:
    # Distance from the classifier's decision boundary. 0.5 = uncertain, 0 or 1 = confident.
    confidence = abs(probability - 0.5) * 2
    return round(float(min(max(confidence, 0.0), 1.0)), 4)


def predict(applicant_dict: dict) -> dict:
    model, shap_explainer, feature_names = _load_artifacts()

    missing = [name for name in feature_names if name not in applicant_dict]
    if missing:
        raise ValueError(f"Applicant data is missing features: {missing}")

    row = pd.DataFrame(
        [{name: applicant_dict[name] for name in feature_names}],
        columns=feature_names,
    )

    probability = float(model.predict_proba(row)[0, 1])
    score = int(probability * 100)
    risk_band = _risk_band(score)
    confidence = _confidence(probability)

    shap_values = shap_explainer.shap_values(row)
    top_factors = explain(shap_values, feature_names, row.iloc[0].to_dict())
    recommended_action = route_decision(score, confidence, risk_band)

    return {
        "score": score,
        "probability": round(probability, 6),
        "risk_band": risk_band,
        "top_factors": top_factors,
        "recommended_action": recommended_action,
        "confidence": confidence,
    }
