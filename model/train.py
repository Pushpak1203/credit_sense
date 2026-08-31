from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    APPLICANTS_DATA_PATH,
    ARTIFACTS_DIR,
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    SHAP_EXPLAINER_PATH,
)

FEATURE_COLUMNS = [
    "monthly_income",
    "repayment_history_score",
    "utility_bill_regularity",
    "years_in_business",
    "debt_to_income_ratio",
    "group_membership",
    "income_stability_score",
]
TARGET_COLUMN = "repaid_on_time"


def train_model() -> dict:
    if not APPLICANTS_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {APPLICANTS_DATA_PATH}. "
            "Run: python data/generate_synthetic_data.py"
        )

    df = pd.read_csv(APPLICANTS_DATA_PATH)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(y_test, predictions),
        "scale_pos_weight": float(scale_pos_weight),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Classification report:")
    print(metrics["classification_report"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # TreeExplainer is deterministic for a trained XGBoost tree ensemble.
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, SHAP_EXPLAINER_PATH)

    with FEATURE_NAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(FEATURE_COLUMNS, file, indent=2)

    return metrics


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
