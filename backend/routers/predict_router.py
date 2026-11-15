from typing import Optional
from controllers.predict_controller import handle_predict
from database.db import get_db
from fastapi import APIRouter, Depends, File, Query, UploadFile
from models.schemas import PredictResponse
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None),
    category: Optional[str] = Query("bottle"),
    db: Session = Depends(get_db),
):
    return await handle_predict(file, db, custom_threshold=threshold, category=category)
