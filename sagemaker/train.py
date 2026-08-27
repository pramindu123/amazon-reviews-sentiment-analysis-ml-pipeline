"""Train a TF-IDF and Logistic Regression sentiment model in SageMaker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument(
        "--train",
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def load_gold_data(input_dir: str | Path) -> pd.DataFrame:
    """Load every Parquet part file downloaded into the SageMaker channel."""
    parquet_files = sorted(Path(input_dir).rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {input_dir}")

    frames = [
        pd.read_parquet(path, columns=["clean_text", "label"])
        for path in parquet_files
    ]
    data = pd.concat(frames, ignore_index=True)

    missing_columns = {"clean_text", "label"} - set(data.columns)
    if missing_columns:
        raise ValueError(f"Gold data is missing columns: {sorted(missing_columns)}")

    data = data.dropna(subset=["clean_text", "label"]).copy()
    data["clean_text"] = data["clean_text"].astype(str).str.strip()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)
    data = data[
        data["clean_text"].ne("") & data["label"].isin([0, 1])
    ].reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid rows remain after validating the Gold data")
    if data["label"].nunique() != 2:
        raise ValueError("Training requires both label classes: 0 and 1")

    return data


def build_model(max_features: int, c_value: float, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    data = load_gold_data(args.train)
    print(f"Loaded {len(data):,} valid Gold records")
    print(f"Label distribution: {data['label'].value_counts().sort_index().to_dict()}")

    label_counts = data["label"].value_counts()
    stratify = data["label"] if label_counts.min() >= 2 else None
    x_train, x_validation, y_train, y_validation = train_test_split(
        data["clean_text"],
        data["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = build_model(args.max_features, args.c, args.random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)

    metrics = {
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(y_validation, predictions, zero_division=0),
        "recall": recall_score(y_validation, predictions, zero_division=0),
        "f1": f1_score(y_validation, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_validation, predictions).tolist(),
        "classification_report": classification_report(
            y_validation,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "training_rows": len(x_train),
        "validation_rows": len(x_validation),
        "scikit_learn_version": sklearn.__version__,
    }

    print(f"validation:accuracy={metrics['accuracy']:.6f}")
    print(f"validation:precision={metrics['precision']:.6f}")
    print(f"validation:recall={metrics['recall']:.6f}")
    print(f"validation:f1={metrics['f1']:.6f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved model artifacts to {model_dir}")
    return metrics


def model_fn(model_dir: str) -> Pipeline:
    """Load the trained model for a future SageMaker endpoint."""
    return joblib.load(Path(model_dir) / "model.joblib")


def input_fn(request_body: str, content_type: str) -> list[str]:
    """Accept either a JSON list or an object containing a texts list."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(request_body)
    texts = payload.get("texts") if isinstance(payload, dict) else payload
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError('Expected JSON such as {"texts": ["great product"]}')
    return texts


def predict_fn(input_data: list[str], model: Pipeline) -> dict[str, Any]:
    predictions = model.predict(input_data).astype(int).tolist()
    probabilities = model.predict_proba(input_data)[:, 1].tolist()
    return {"predictions": predictions, "positive_probabilities": probabilities}


def output_fn(prediction: dict[str, Any], accept: str) -> tuple[str, str]:
    if accept not in {"application/json", "*/*"}:
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction), "application/json"


if __name__ == "__main__":
    train(parse_args())
