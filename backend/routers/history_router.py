from typing import List

from controllers.history_controller import handle_history
from database.db import get_db
from fastapi import APIRouter, Depends
from models.schemas import HistoryItem
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/history", response_model=List[HistoryItem])
def history(db: Session = Depends(get_db)):
    return handle_history(db)
