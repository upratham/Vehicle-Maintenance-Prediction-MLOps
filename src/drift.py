"""Drift monitoring: Population Stability Index + prediction-log store.

Training distribution is persisted once at transformation time
(artifact/training_distribution.json). Live predictions are appended to a
MongoDB collection (`prediction_logs`). PSI is computed per feature by
bucketing the live window against the training histogram.

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


def _resolve_training_dist_path() -> str:
    """Find training_distribution.json in the latest pipeline run dir."""
    pipeline_base = os.path.join("artifact", "vehicle_maintenance")
    if os.path.isdir(pipeline_base):
        runs = sorted(
            [d for d in os.listdir(pipeline_base) if os.path.isdir(os.path.join(pipeline_base, d))],
            reverse=True,
        )
        for run in runs:
            candidate = os.path.join(pipeline_base, run, "training_distribution.json")
            if os.path.exists(candidate):
                return candidate
    return os.path.join("artifact", "training_distribution.json")


def _load_training_distribution() -> dict[str, Any] | None:
    path = _resolve_training_dist_path()
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


def log_prediction(payload: dict[str, Any], response: dict[str, Any]) -> None:
    col = _mongo_collection()
    if col is None:
        return
    try:
        col.insert_one({
            "ts": datetime.now(timezone.utc),
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


def compute_drift(window: str = "7d") -> dict[str, Any]:
    """Compute per-feature PSI of the live window vs. training distribution.

    Falls back to a synthetic "mildly in-distribution" demo report if Mongo
    has no logs yet, so the Ops page always renders.
    """
    training = _load_training_distribution()
    if training is None:
        return {"window": window, "source": "none", "features": [], "error": "training distribution not found"}

    delta = _parse_window(window)
    since = datetime.now(timezone.utc) - delta

    col = _mongo_collection()
    rows: list[dict[str, Any]] = []
    source = "mongo"
    if col is not None:
        try:
            rows = list(col.find({"ts": {"$gte": since}}, {"_id": 0, "input": 1}))
        except Exception as exc:
            logging.warning(f"prediction_log query failed: {exc}")
            rows = []

    if not rows:
        source = "demo"
        # Seed a demo window so the chart is populated for a fresh deploy.
        rows = _demo_rows()

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
        "source": source,
        "n_predictions": len(rows),
        "thresholds": {"stable": 0.1, "warning": 0.25},
        "features": features,
    }


def _demo_rows() -> list[dict[str, Any]]:
    """Synthetic in-distribution window used when no live logs exist yet."""
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(120):
        rows.append({"input": {
            "Reported_Issues": int(rng.integers(0, 5)),
            "Vehicle_Age": int(rng.integers(1, 14)),
            "Engine_Size": round(float(rng.normal(2.4, 0.8)), 2),
            "Odometer_Reading": int(rng.normal(95_000, 40_000)),
            "Accident_History": int(rng.integers(0, 3)),
            "Fuel_Efficiency": round(float(rng.normal(14.0, 3.5)), 2),
            "Tire_Condition": str(rng.choice(["Worn Out", "Good", "New"], p=[0.3, 0.45, 0.25])),
            "Brake_Condition": str(rng.choice(["Worn Out", "Good", "New"], p=[0.28, 0.48, 0.24])),
            "Battery_Status": str(rng.choice(["Weak", "Good", "Strong"], p=[0.3, 0.43, 0.27])),
            "Vehicle_Model": str(rng.choice(["Car", "SUV", "Truck", "Van", "Bus", "Motorcycle"],
                                            p=[0.34, 0.22, 0.15, 0.11, 0.10, 0.08])),
            "Fuel_Type": str(rng.choice(["Petrol", "Diesel", "Electric"], p=[0.58, 0.30, 0.12])),
            "Transmission_Type": str(rng.choice(["Automatic", "Manual"], p=[0.66, 0.34])),
        }})
    return rows
