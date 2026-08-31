from fastapi import APIRouter

from api.schemas import FeedbackInput, FeedbackResponse
from feedback.feedback_loop import log_outcome, retrain_if_sufficient

router = APIRouter(tags=["Feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackInput) -> FeedbackResponse:
    log_outcome(feedback.applicant_id, feedback.actual_outcome)
    retrained = retrain_if_sufficient()

    message = "Feedback outcome recorded successfully."
    if retrained:
        message += " Retraining was triggered after reaching the feedback threshold."

    return FeedbackResponse(message=message)
