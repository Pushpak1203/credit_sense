from typing import Literal
from pydantic import BaseModel, Field


class ApplicantInput(BaseModel):
    monthly_income: float = Field(ge=5000, le=100000)
    repayment_history_score: float = Field(ge=0, le=1)
    utility_bill_regularity: float = Field(ge=0, le=1)
    years_in_business: int = Field(ge=0, le=20)
    debt_to_income_ratio: float = Field(ge=0, le=0.9)
    group_membership: int = Field(ge=0, le=1)
    income_stability_score: float = Field(ge=0, le=1)


class Factor(BaseModel):
    feature: str
    impact: Literal["positive", "negative"]
    plain_english: str


class ScoreResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    probability: float = Field(ge=0, le=1)
    risk_band: Literal["Low", "Medium", "High"]
    top_factors: list[Factor]
    recommended_action: Literal["auto_approve", "auto_reject", "manual_review"]
    confidence: float = Field(ge=0, le=1)
    applicant_id: str


class FeedbackInput(BaseModel):
    applicant_id: str = Field(min_length=1, max_length=128)
    actual_outcome: int = Field(ge=0, le=1)


class FeedbackResponse(BaseModel):
    message: str
