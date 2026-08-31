# Credit Sense

Credit Sense is an AI-assisted credit-decisioning system designed for Indian NBFC use cases involving underbanked borrowers such as farmers, gig workers, and small business owners. It uses alternative financial signals to estimate the probability of timely repayment with an XGBoost model and explains the strongest contributing factors with SHAP. The system routes decisions to auto-approve, auto-reject, or manual review based on configurable score and confidence thresholds.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate training data:

```bash
python data/generate_synthetic_data.py
```

Train the model and generate artifacts:

```bash
python model/train.py
```

Run the test suite:

```bash
pytest -q
```

## Run the API

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Score an applicant

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/score" \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_income": 55000,
    "repayment_history_score": 0.82,
    "utility_bill_regularity": 0.91,
    "years_in_business": 6,
    "debt_to_income_ratio": 0.28,
    "group_membership": 1,
    "income_stability_score": 0.88
  }'
```

The response contains a server-generated `applicant_id`. Keep it for outcome feedback.

## Submit repayment feedback

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "replace-with-applicant-id",
    "actual_outcome": 1
  }'
```

## Project structure

```text
credit_sense/
├── data/
│   ├── __init__.py
│   └── generate_synthetic_data.py
├── model/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   └── artifacts/
├── explainer/
│   ├── __init__.py
│   └── shap_explainer.py
├── router/
│   ├── __init__.py
│   └── decision_router.py
├── feedback/
│   ├── __init__.py
│   └── feedback_loop.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── __init__.py
│       ├── score.py
│       └── feedback.py
├── config/
│   ├── __init__.py
│   └── config.py
├── tests/
│   ├── test_model.py
│   ├── test_router.py
│   └── test_api.py
├── requirements.txt
├── README.md
└── .env.example
```

## Decisioning and human-in-the-loop design

The score is `int(probability * 100)`. Scores of 65 or above are Low risk, 40–64 are Medium risk, and below 40 are High risk. Auto-approval requires a score of at least 65 and confidence of at least 0.75; auto-rejection requires a score below 40 and confidence of at least 0.75; all other cases go to manual review.

Confidence is derived from the model probability's distance from the 0.5 decision boundary. This is a practical routing heuristic, not a calibrated regulatory-grade uncertainty estimate. For production deployment, probability calibration, bias testing, drift monitoring, audit logging, consent management, data lineage, adverse-action compliance, and human override controls should be added before making real lending decisions.

The feedback endpoint logs observed outcomes and triggers retraining once the configured number of new feedback rows is reached. This implementation retrains on the current synthetic training dataset because the API only receives applicant IDs and outcomes; a real production system must persist the original feature snapshot for each scored applicant and join it with verified outcomes before retraining.
