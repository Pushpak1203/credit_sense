from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter

from api.schemas import ApplicantInput, ScoreResponse
from model.predict import predict

router = APIRouter(tags=["Scoring"])


@router.post("/score", response_model=ScoreResponse)
def score_applicant(applicant: ApplicantInput) -> ScoreResponse:
    applicant_id = str(uuid4())
    result = predict(applicant.model_dump())

    print(
        f"[{datetime.now(timezone.utc).isoformat()}] "
        f"Scored applicant_id={applicant_id} score={result['score']}"
    )

    return ScoreResponse(applicant_id=applicant_id, **result)
