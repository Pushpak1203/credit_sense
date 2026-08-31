from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import FEEDBACK_LOG_PATH, RETRAIN_STATE_PATH
from model.train import train_model


def log_outcome(applicant_id: str, actual_outcome: int) -> None:
    if actual_outcome not in (0, 1):
        raise ValueError("actual_outcome must be 0 or 1")

    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "applicant_id": str(applicant_id),
                "actual_outcome": int(actual_outcome),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    row.to_csv(
        FEEDBACK_LOG_PATH,
        mode="a",
        header=not FEEDBACK_LOG_PATH.exists(),
        index=False,
    )


def _last_retrain_count() -> int:
    if not RETRAIN_STATE_PATH.exists():
        return 0
    try:
        with RETRAIN_STATE_PATH.open("r", encoding="utf-8") as file:
            return int(json.load(file).get("feedback_count_at_last_retrain", 0))
    except (json.JSONDecodeError, OSError, ValueError):
        return 0


def retrain_if_sufficient(threshold: int = 100) -> bool:
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    if not FEEDBACK_LOG_PATH.exists():
        return False

    feedback_count = len(pd.read_csv(FEEDBACK_LOG_PATH))
    new_feedback_count = feedback_count - _last_retrain_count()
    if new_feedback_count < threshold:
        return False

    train_model()
    with RETRAIN_STATE_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "feedback_count_at_last_retrain": feedback_count,
                "retrained_at": datetime.now(timezone.utc).isoformat(),
            },
            file,
            indent=2,
        )
    return True
