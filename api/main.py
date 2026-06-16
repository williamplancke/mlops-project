import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column, DateTime, Integer, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.model_artifacts import MODEL_FILES

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
DAMAGE_THRESHOLD = float(os.getenv("DAMAGE_THRESHOLD", "0.5"))


def database_url() -> str:
    if explicit_url := os.getenv("DATABASE_URL"):
        return explicit_url

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    if all([user, password, database]):
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"

    return "sqlite:///./predictions.db"


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    request = Column(JSON, nullable=False)
    response = Column(JSON, nullable=False)


class PredictionRequest(BaseModel):
    features: dict[str, float | int | bool | None] = Field(
        ..., description="Customer feature values keyed by training column name."
    )


class PredictionResponse(BaseModel):
    predicted_profit: float
    damage_probability: float
    predicted_damage_incident: bool
    predicted_damage_amount_if_damage: float
    expected_damage_amount: float
    expected_net_profit: float
    missing_features_filled: list[str]


class ModelBundle:
    def __init__(self, model_dir: Path):
        metadata_path = model_dir / MODEL_FILES["metadata"]
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing {metadata_path}. Run training/train.py before starting the API."
            )

        self.metadata = json.loads(metadata_path.read_text())
        self.feature_columns = self.metadata["feature_columns"]
        self.feature_defaults = self.metadata["feature_defaults"]
        self.profit_model = joblib.load(model_dir / MODEL_FILES["profit"])
        self.damage_incidence_model = joblib.load(
            model_dir / MODEL_FILES["damage_incidence"]
        )
        self.damage_amount_model = joblib.load(
            model_dir / MODEL_FILES["damage_amount"]
        )

    def frame_from_payload(
        self, payload: dict[str, Any]
    ) -> tuple[pd.DataFrame, list[str]]:
        unknown = sorted(set(payload) - set(self.feature_columns))
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown feature columns: {unknown}",
            )

        row = {}
        missing = []
        for column in self.feature_columns:
            value = payload.get(column)
            if value is None:
                row[column] = self.feature_defaults.get(column, 0)
                missing.append(column)
            else:
                row[column] = value

        return pd.DataFrame([row], columns=self.feature_columns), missing

    def predict(self, payload: dict[str, Any]) -> PredictionResponse:
        frame, missing = self.frame_from_payload(payload)
        profit = float(self.profit_model.predict(frame)[0])
        damage_probability = float(
            self.damage_incidence_model.predict_proba(frame)[0, 1]
        )
        predicted_damage_incident = damage_probability >= DAMAGE_THRESHOLD
        if predicted_damage_incident:
            amount_if_damage = float(
                np.expm1(self.damage_amount_model.predict(frame)[0])
            )
            amount_if_damage = max(amount_if_damage, 0.0)
            expected_damage = damage_probability * amount_if_damage
        else:
            amount_if_damage = 0.0
            expected_damage = 0.0

        return PredictionResponse(
            predicted_profit=round(profit, 2),
            damage_probability=round(damage_probability, 4),
            predicted_damage_incident=predicted_damage_incident,
            predicted_damage_amount_if_damage=round(amount_if_damage, 2),
            expected_damage_amount=round(expected_damage, 2),
            expected_net_profit=round(profit - expected_damage, 2),
            missing_features_filled=missing,
        )


def create_db_session() -> sessionmaker[Session]:
    url = database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelBundle(MODEL_DIR)
    app.state.session_factory = create_db_session()
    yield


app = FastAPI(
    title="Customer Value Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


def model_bundle() -> ModelBundle:
    if not hasattr(app.state, "models"):
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")
    return app.state.models


def session_factory() -> sessionmaker[Session]:
    if not hasattr(app.state, "session_factory"):
        raise HTTPException(status_code=503, detail="Database is not ready yet.")
    return app.state.session_factory


@app.get("/health")
def health() -> dict[str, str]:
    with session_factory()() as session:
        session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/features")
def features() -> dict[str, Any]:
    models = model_bundle()
    return {
        "features": models.feature_columns,
        "defaults": models.feature_defaults,
        "metrics": models.metadata.get("metrics", {}),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    response = model_bundle().predict(request.features)
    with session_factory()() as session:
        session.add(
            PredictionLog(
                created_at=datetime.now(timezone.utc),
                request=request.model_dump(mode="json"),
                response=response.model_dump(mode="json"),
            )
        )
        session.commit()
    return response
