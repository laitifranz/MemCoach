from pydantic import BaseModel


class ScoreResponse(BaseModel):
    score: float
    latency_ms: float


class ScoreFeedbackResponse(ScoreResponse):
    feedback: str
    feedback_latency_ms: float
