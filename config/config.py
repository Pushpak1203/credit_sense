from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

AUTO_APPROVE_SCORE_THRESHOLD = 65
AUTO_REJECT_SCORE_THRESHOLD = 40
CONFIDENCE_THRESHOLD = 0.75

MODEL_PATH = ARTIFACTS_DIR / "credit_sense_model.joblib"
SHAP_EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"

FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.csv"
RETRAIN_STATE_PATH = DATA_DIR / "retrain_state.json"
APPLICANTS_DATA_PATH = DATA_DIR / "applicants.csv"

RISK_BANDS = {
    "Low": (65, 100),
    "Medium": (40, 64),
    "High": (0, 39),
}
