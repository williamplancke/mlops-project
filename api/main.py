import json
import os
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


MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")
DAMAGE_THRESHOLD = float(os.getenv("DAMAGE_THRESHOLD", "0.5"))


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
        metadata_path = model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing {metadata_path}. Run training/train.py before starting the API."
            )

        self.metadata = json.loads(metadata_path.read_text())
        self.feature_columns = self.metadata["feature_columns"]
        self.feature_defaults = self.metadata["feature_defaults"]
        self.profit_model = joblib.load(model_dir / "profit_model.joblib")
        self.damage_incidence_model = joblib.load(
            model_dir / "damage_incidence_model.joblib"
        )
        self.damage_amount_model = joblib.load(model_dir / "damage_amount_model.joblib")

    def frame_from_payload(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
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
            amount_if_damage = float(np.expm1(self.damage_amount_model.predict(frame)[0]))
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
    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


app = FastAPI(title="Customer Value Prediction API", version="1.0.0")
SessionLocal = create_db_session()
models: ModelBundle | None = None


@app.on_event("startup")
def load_models() -> None:
    global models
    models = ModelBundle(MODEL_DIR)


@app.get("/health")
def health() -> dict[str, str]:
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/features")
def features() -> dict[str, Any]:
    if models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")
    return {
        "features": models.feature_columns,
        "defaults": models.feature_defaults,
        "metrics": models.metadata.get("metrics", {}),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    response = models.predict(request.features)
    with SessionLocal() as session:
        session.add(
            PredictionLog(
                created_at=datetime.now(timezone.utc),
                request=request.model_dump(mode="json"),
                response=response.model_dump(mode="json"),
            )
        )
        session.commit()
    return response
