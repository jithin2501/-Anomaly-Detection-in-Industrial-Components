from datetime import datetime

from database.db import Base
from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    label: Mapped[str] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
