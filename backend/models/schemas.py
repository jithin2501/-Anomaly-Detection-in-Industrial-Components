from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PredictResponse(BaseModel):
    label: str          # "NORMAL" or "DEFECTIVE"
    score: float
    threshold: float
    heatmap: str        # base64 PNG


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    label: str
    score: float
    threshold: float
    created_at: datetime
