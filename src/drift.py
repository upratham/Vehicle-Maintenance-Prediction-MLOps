"""Drift monitoring: Population Stability Index + prediction-log store.

Training distribution is persisted once at transformation time
(artifact/<profile>/<run>/training_distribution.json). Live predictions are
appended to a MongoDB collection (`prediction_logs`) tagged with a
`profile_name` field. PSI is computed per feature by bucketing the live
window against the training histogram.

All Mongo access is lazy and best-effort — the web app must not fail if
Mongo is unavailable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from src.logger import logging

PREDICTION_LOG_COLLECTION = "prediction_logs"


def _resolve_training_dist_path(profile_name: str = "vehicle_maintenance") -> str:
    pipeline_base = os.path.join("artifact", profile_name)
    if os.path.isdir(pipeline_base):
        runs = sorted(
            [d for d in os.listdir(pipeline_base) if os.path.isdir(os.path.join(pipeline_base, d))],
            reverse=True,
        )
        for run in runs:
            candidate = os.path.join(pipeline_base, run, "training_distribution.json")
            if os.path.exists(candidate):
                return candidate
    return os.path.join("artifact", f"{profile_name}_training_distribution.json")


def _load_training_distribution(profile_name: str = "vehicle_maintenance") -> dict[str, Any] | None:
    path = _resolve_training_dist_path(profile_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _mongo_collection():
    """Return the prediction-log collection or None if Mongo isn't reachable."""
    try:
        from pymongo import MongoClient

        url = os.getenv("CONNECTION_URL")
        db_name = os.getenv("DB_USERNAME") or "vehicle_maintenance"
        if not url:
            return None
        client = MongoClient(url, serverSelectionTimeoutMS=1500)
        return client[db_name][PREDICTION_LOG_COLLECTION]
    except Exception as exc:
        logging.warning(f"Mongo unavailable for drift logging: {exc}")
        return None


def log_prediction(
    payload: dict[str, Any],
    response: dict[str, Any],
    profile_name: str = "vehicle_maintenance",
) -> None:
    col = _mongo_collection()
    if col is None:
        return
    try:
        col.insert_one({
            "ts": datetime.now(timezone.utc),
            "profile_name": profile_name,
            "input": payload,
            "score": response.get("score"),
            "label": response.get("label"),
            "status": response.get("status"),
        })
    except Exception as exc:
        logging.warning(f"prediction_log insert failed: {exc}")


def _parse_window(window: str) -> timedelta:
    window = (window or "7d").strip().lower()
    if window.endswith("d"):
        return timedelta(days=int(window[:-1] or 7))
    if window.endswith("h"):
        return timedelta(hours=int(window[:-1] or 24))
    return timedelta(days=7)


def _psi_numeric(train_bins: list[float], train_counts: list[int], live: list[float]) -> float:
    train_counts_arr = np.asarray(train_counts, dtype=float)
    train_total = train_counts_arr.sum() or 1.0
    train_pct = train_counts_arr / train_total

    live_arr = np.asarray(live, dtype=float)
    live_counts, _ = np.histogram(live_arr, bins=train_bins)
    live_total = live_counts.sum() or 1.0
    live_pct = live_counts / live_total

    eps = 1e-4
    train_pct = np.clip(train_pct, eps, None)
    live_pct = np.clip(live_pct, eps, None)
    return float(np.sum((live_pct - train_pct) * np.log(live_pct / train_pct)))


def _psi_categorical(train_counts: dict[str, int], live: list[str]) -> float:
    categories = list(train_counts.keys())
    train_total = sum(train_counts.values()) or 1
    train_pct = np.array([train_counts[c] / train_total for c in categories], dtype=float)

    live_counts = {c: 0 for c in categories}
    for v in live:
        if v in live_counts:
            live_counts[v] += 1
    live_total = sum(live_counts.values()) or 1
    live_pct = np.array([live_counts[c] / live_total for c in categories], dtype=float)

    eps = 1e-4
    train_pct = np.clip(train_pct, eps, None)
    live_pct = np.clip(live_pct, eps, None)
    return float(np.sum((live_pct - train_pct) * np.log(live_pct / train_pct)))


def _verdict(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "warning"
    return "drifted"


def _demo_rows(training: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate 120 in-distribution demo rows from the training distribution stats."""
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(120):
        inp: dict[str, Any] = {}
        for feat, meta in training.get("numerical", {}).items():
            mean = float(meta.get("mean", 0))
            std = float(meta.get("std", max(abs(mean) * 0.1, 1.0)))
            inp[feat] = round(float(rng.normal(mean, std)), 4)
        for feat, counts in training.get("categorical", {}).items():
            categories = list(counts.keys())
            total = sum(counts.values()) or 1
            probs = [counts[c] / total for c in categories]
            inp[feat] = str(rng.choice(categories, p=probs))
        rows.append({"input": inp})
    return rows


def compute_drift(
    window: str = "7d",
    profile_name: str = "vehicle_maintenance",
) -> dict[str, Any]:
    """Compute per-feature PSI of the live window vs. training distribution for a profile."""
    training = _load_training_distribution(profile_name)
    if training is None:
        return {
            "window": window,
            "profile": profile_name,
            "source": "none",
            "features": [],
            "error": "training distribution not found",
        }

    delta = _parse_window(window)
    since = datetime.now(timezone.utc) - delta

    col = _mongo_collection()
    rows: list[dict[str, Any]] = []
    source = "mongo"
    if col is not None:
        try:
            rows = list(col.find(
                {"ts": {"$gte": since}, "profile_name": profile_name},
                {"_id": 0, "input": 1},
            ))
        except Exception as exc:
            logging.warning(f"prediction_log query failed: {exc}")
            rows = []

    if not rows:
        source = "demo"
        rows = _demo_rows(training)

    features: list[dict[str, Any]] = []

    for feat, meta in training.get("numerical", {}).items():
        live_values = [float(r["input"].get(feat, meta["mean"])) for r in rows if feat in r.get("input", {})]
        if not live_values:
            continue
        psi = _psi_numeric(meta["bins"], meta["counts"], live_values)
        features.append({
            "feature": feat,
            "type": "numeric",
            "psi": round(psi, 4),
            "verdict": _verdict(psi),
            "n_live": len(live_values),
        })

    for feat, counts in training.get("categorical", {}).items():
        live_values = [str(r["input"].get(feat, "")) for r in rows if feat in r.get("input", {})]
        live_values = [v for v in live_values if v]
        if not live_values:
            continue
        psi = _psi_categorical(counts, live_values)
        features.append({
            "feature": feat,
            "type": "categorical",
            "psi": round(psi, 4),
            "verdict": _verdict(psi),
            "n_live": len(live_values),
        })

    features.sort(key=lambda f: f["psi"], reverse=True)
    return {
        "window": window,
        "profile": profile_name,
        "source": source,
        "n_predictions": len(rows),
        "thresholds": {"stable": 0.1, "warning": 0.25},
        "features": features,
    }
