import hashlib
import json
import logging
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
log = logging.getLogger(__name__)
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()


def validate_rows(rows, day):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Empty or invalid rows")
    seen = set()
    for index, row in enumerate(rows):
        customer_id = row["customer_id"]
        if not isinstance(customer_id, str) or not customer_id.strip():
            raise ValueError(f"Row {index}: customer_id must be a non-empty string")
        if customer_id in seen:
            raise ValueError(f"Row {index}: duplicate customer_id {customer_id!r}")
        seen.add(customer_id)
        if row["business_date"] != day:
            raise ValueError(f"Row {index} ({customer_id}): business_date {row['business_date']!r} != {day!r}")
        for name, (low, high) in zip(FEATURES, LIMITS):
            value = row[name]
            # не isinstance: bool тоже int
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Row {index} ({customer_id}): {name}={value!r} is not a finite number")
            if not low <= value <= high:
                raise ValueError(f"Row {index} ({customer_id}): {name}={value!r} outside [{low}, {high}]")
    return rows


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    # True и 2.5 тоже сюда
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError(f"batch_size must be an integer in [1, 1000], got {batch_size!r}")
    predictions = []
    for number, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[start:start + batch_size]
        log.info("Batch %d: %d rows", number, len(batch))
        predictions.extend(predict(batch, model))
    return predictions


def validate_predictions(predictions, expected_ids):
    expected = list(expected_ids)
    if len(predictions) != len(expected):
        raise ValueError(f"Expected {len(expected)} predictions, got {len(predictions)}")
    seen = set()
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict) or not {"customer_id", "score", "label"} <= prediction.keys():
            raise ValueError(f"Prediction {index}: missing customer_id/score/label")
        customer_id, score, label = prediction["customer_id"], prediction["score"], prediction["label"]
        if customer_id in seen:
            raise ValueError(f"Prediction {index}: duplicate customer_id {customer_id!r}")
        seen.add(customer_id)
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Prediction {index} ({customer_id}): invalid score {score!r}")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Prediction {index} ({customer_id}): invalid label {label!r}")
        if label != int(score >= 0.5):
            raise ValueError(f"Prediction {index} ({customer_id}): label {label} does not match score {score}")
    if seen != set(expected):
        missing, extra = set(expected) - seen, seen - set(expected)
        raise ValueError(f"customer_id mismatch: missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")


def replace_day(conn, day, model_version, predictions):
    # проверяем до DELETE, чтобы не затереть старый день
    if not predictions:
        raise ValueError("Empty predictions")
    values = [(day, model_version, p["customer_id"], p["score"], p["label"]) for p in predictions]
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (day + ":" + model_version,))
        cur.execute("DELETE FROM predictions WHERE business_date = %s AND model_version = %s",
                    (day, model_version))
        cur.executemany(
            "INSERT INTO predictions (business_date, model_version, customer_id, score, label) "
            "VALUES (%s, %s, %s, %s, %s)",
            values,
        )
