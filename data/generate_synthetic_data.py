from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import APPLICANTS_DATA_PATH


def generate_synthetic_data(n_records: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    monthly_income = rng.uniform(5000, 100000, n_records)
    repayment_history_score = rng.uniform(0, 1, n_records)
    utility_bill_regularity = rng.uniform(0, 1, n_records)
    years_in_business = rng.integers(0, 21, n_records)
    debt_to_income_ratio = rng.uniform(0.0, 0.9, n_records)
    group_membership = rng.integers(0, 2, n_records)
    income_stability_score = rng.uniform(0, 1, n_records)

    deterministic_outcome = (
        (repayment_history_score > 0.55)
        & (utility_bill_regularity > 0.45)
        & (debt_to_income_ratio < 0.5)
    ).astype(int)

    noise_mask = rng.random(n_records) < 0.10
    repaid_on_time = deterministic_outcome.copy()
    repaid_on_time[noise_mask] = 1 - repaid_on_time[noise_mask]

    return pd.DataFrame(
        {
            "monthly_income": monthly_income.round(2),
            "repayment_history_score": repayment_history_score.round(4),
            "utility_bill_regularity": utility_bill_regularity.round(4),
            "years_in_business": years_in_business,
            "debt_to_income_ratio": debt_to_income_ratio.round(4),
            "group_membership": group_membership,
            "income_stability_score": income_stability_score.round(4),
            "repaid_on_time": repaid_on_time,
        }
    )


def main() -> None:
    APPLICANTS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(APPLICANTS_DATA_PATH, index=False)
    print(f"Generated {len(df)} synthetic applicant records at {APPLICANTS_DATA_PATH}")


if __name__ == "__main__":
    main()
