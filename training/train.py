import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.model_artifacts import MODEL_FILES, TARGET_COLUMNS

RANDOM_STATE = 304
DEFAULT_QUICK_ROWS = 2000


def maybe_quick(params: dict, quick: bool) -> dict:
    if not quick:
        return params
    updated = params.copy()
    updated["n_estimators"] = min(updated.get("n_estimators", 100), 15)
    updated["max_depth"] = min(updated.get("max_depth", 3), 3)
    return updated


def quick_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)


def split_features_target(
    df: pd.DataFrame, target: str, drop_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    data = df.drop(columns=drop_columns)
    return data.drop(columns=[target]), data[target]


def save_model(model, output_dir: Path, filename: str) -> None:
    joblib.dump(model, output_dir / filename)


def train_profit(df: pd.DataFrame, output_dir: Path, quick: bool) -> dict:
    print("Training profit model...", flush=True)
    x, y = split_features_target(
        df,
        target="outcome_profit",
        drop_columns=["outcome_damage_amount", "outcome_damage_inc"],
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    params = maybe_quick(
        {
            "criterion": "squared_error",
            "learning_rate": 0.12,
            "loss": "squared_error",
            "max_depth": 3,
            "max_features": None,
            "max_leaf_nodes": 54,
            "min_samples_leaf": 3,
            "min_samples_split": 14,
            "n_estimators": 493,
            "subsample": 0.87,
        },
        quick,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gbm", GradientBoostingRegressor(**params, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }

    model.fit(x, y)
    save_model(model, output_dir, MODEL_FILES["profit"])
    return metrics


def train_damage_incidence(df: pd.DataFrame, output_dir: Path, quick: bool) -> dict:
    print("Training damage incidence model...", flush=True)
    x, y = split_features_target(
        df,
        target="outcome_damage_inc",
        drop_columns=["outcome_damage_amount", "outcome_profit"],
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    params = maybe_quick(
        {
            "loss": "exponential",
            "learning_rate": 0.25,
            "n_estimators": 484,
            "subsample": 0.51,
            "criterion": "friedman_mse",
            "min_samples_split": 4,
            "min_samples_leaf": 3,
            "max_depth": 12,
            "tol": 0.0449,
        },
        quick,
    )

    base_pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=0.8, random_state=42)),
            ("under", RandomUnderSampler(random_state=67)),
            ("gbc", GradientBoostingClassifier(**params, random_state=42)),
        ]
    )
    cv = 2 if quick else 5
    model = CalibratedClassifierCV(
        estimator=base_pipeline,
        cv=cv,
        method="sigmoid" if quick else "isotonic",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "brier": float(brier_score_loss(y_test, prob)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }

    model.fit(x, y)
    save_model(model, output_dir, MODEL_FILES["damage_incidence"])
    return metrics


def train_damage_amount(df: pd.DataFrame, output_dir: Path, quick: bool) -> dict:
    print("Training damage amount model...", flush=True)
    data = df[df["outcome_damage_inc"] != 0].copy()
    x, y = split_features_target(
        data,
        target="outcome_damage_amount",
        drop_columns=["outcome_profit", "outcome_damage_inc"],
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1234
    )
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    params = maybe_quick(
        {
            "n_estimators": 397,
            "learning_rate": 0.03,
            "max_depth": 12,
            "max_leaf_nodes": 32,
            "min_samples_split": 2,
            "min_samples_leaf": 16,
            "max_features": "log2",
            "subsample": 0.737,
            "loss": "squared_error",
            "criterion": "squared_error",
        },
        quick,
    )

    model = Pipeline(
        [("gbm", GradientBoostingRegressor(**params, random_state=42))]
    )
    model.fit(x_train, y_train_log)
    pred_log = model.predict(x_test)
    pred = np.expm1(pred_log)

    metrics = {
        "r2_log": float(r2_score(y_test_log, pred_log)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }

    model.fit(x, np.log1p(y))
    save_model(model, output_dir, MODEL_FILES["damage_amount"])
    return metrics


def write_metadata(df: pd.DataFrame, output_dir: Path, metrics: dict) -> None:
    feature_columns = [column for column in df.columns if column not in TARGET_COLUMNS]
    defaults = df[feature_columns].median(numeric_only=True).fillna(0).to_dict()
    metadata = {
        "target_columns": TARGET_COLUMNS,
        "feature_columns": feature_columns,
        "feature_defaults": defaults,
        "metrics": metrics,
    }
    (output_dir / MODEL_FILES["metadata"]).write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train_V2_cleaned.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--quick-rows",
        type=int,
        default=DEFAULT_QUICK_ROWS,
        help="Maximum rows used when --quick is enabled.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)

    missing_targets = [target for target in TARGET_COLUMNS if target not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing required target columns: {missing_targets}")

    if args.quick:
        df = quick_sample(df, args.quick_rows)
        print(f"Quick mode enabled: using {len(df)} rows.", flush=True)

    metrics = {
        "profit": train_profit(df, output_dir, args.quick),
        "damage_incidence": train_damage_incidence(df, output_dir, args.quick),
        "damage_amount": train_damage_amount(df, output_dir, args.quick),
    }
    write_metadata(df, output_dir, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
