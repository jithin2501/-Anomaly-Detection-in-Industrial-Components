from contextlib import asynccontextmanager

import config
from database.db import Base, engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import db_models  # noqa: F401  (registers tables)
from routers import history_router, models_router, predict_router
from services import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    inference_service.load()          # model loaded once, not per request
    yield


app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict_router.router)
app.include_router(history_router.router)
app.include_router(models_router.router)


@app.get("/health")
def health():
    return {"status": "ok"}
