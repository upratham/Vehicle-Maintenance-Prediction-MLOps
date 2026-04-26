"""Rule-based cost estimation and feature-impact explanation.

Kept deliberately simple and deterministic so the demo is reliable:
  - Cost/hours are looked up from the MongoDB `repair_costs` collection.
  - Falls back to hardcoded data if MongoDB is unreachable.
  - Feature impact is a heuristic "contribution score" computed from the user's
    inputs; it mirrors what a SHAP plot would show but has no external deps and
    can't fail at demo time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.logger import logging

_REPAIR_COSTS_COLLECTION = "repair_costs"

_HARDCODED_COSTS = [
    {"trigger": "brake_worn",          "service": "Brake pad replacement",           "cost_low": 180,  "cost_high": 450,  "hours_low": 1.0, "hours_high": 2.5, "notes": "Front or rear pads; includes labor"},
    {"trigger": "brake_worn_severe",   "service": "Brake rotor + pad service",       "cost_low": 350,  "cost_high": 800,  "hours_low": 2.0, "hours_high": 4.0, "notes": "When rotors also need resurfacing or replacement"},
    {"trigger": "battery_weak",        "service": "Battery replacement",             "cost_low": 120,  "cost_high": 320,  "hours_low": 0.3, "hours_high": 1.0, "notes": "Standard 12V; premium/AGM higher"},
    {"trigger": "tire_worn",           "service": "Tire replacement (set of 4)",     "cost_low": 500,  "cost_high": 1400, "hours_low": 1.0, "hours_high": 2.0, "notes": "Includes mounting and balancing"},
    {"trigger": "tire_mild",           "service": "Tire rotation & alignment",       "cost_low": 80,   "cost_high": 180,  "hours_low": 0.5, "hours_high": 1.5, "notes": "Preventive maintenance"},
    {"trigger": "high_mileage_routine","service": "Oil & filter change",             "cost_low": 40,   "cost_high": 110,  "hours_low": 0.5, "hours_high": 1.0, "notes": "Conventional to full synthetic"},
    {"trigger": "high_mileage_trans",  "service": "Transmission fluid service",      "cost_low": 180,  "cost_high": 400,  "hours_low": 1.0, "hours_high": 2.0, "notes": "Manual or automatic"},
    {"trigger": "engine_age",          "service": "Spark plug replacement",          "cost_low": 120,  "cost_high": 380,  "hours_low": 1.0, "hours_high": 2.5, "notes": "Set of 4-8 depending on engine"},
    {"trigger": "routine",             "service": "Air filter replacement",          "cost_low": 25,   "cost_high": 90,   "hours_low": 0.3, "hours_high": 0.8, "notes": "Engine or cabin filter"},
    {"trigger": "engine_age_high",     "service": "Timing belt service",             "cost_low": 450,  "cost_high": 1200, "hours_low": 3.0, "hours_high": 6.0, "notes": "Major interval service"},
    {"trigger": "battery_weak_severe", "service": "Alternator replacement",          "cost_low": 400,  "cost_high": 900,  "hours_low": 1.5, "hours_high": 3.5, "notes": "If battery drains persist"},
    {"trigger": "age_high",            "service": "Suspension inspection",           "cost_low": 80,   "cost_high": 200,  "hours_low": 0.5, "hours_high": 1.5, "notes": "Diagnostic only"},
    {"trigger": "multiple_flags",      "service": "Full inspection (multi-point)",   "cost_low": 100,  "cost_high": 250,  "hours_low": 1.0, "hours_high": 2.0, "notes": "Recommended when several signals flag"},
    {"trigger": "default",             "service": "General service & diagnostic",    "cost_low": 90,   "cost_high": 260,  "hours_low": 0.8, "hours_high": 2.0, "notes": "Fallback estimate"},
]


@dataclass
class ServiceEstimate:
    service: str
    cost_low: float
    cost_high: float
    hours_low: float
    hours_high: float
    notes: str


@dataclass
class FeatureImpact:
    feature: str        # display label
    value: Any          # user's entered value
    contribution: float # signed score in ~[-1, 1]; positive = pushes toward "maintenance"


def _rows_to_dict(rows: list[dict]) -> dict[str, ServiceEstimate]:
    return {
        r["trigger"]: ServiceEstimate(
            service=r["service"],
            cost_low=float(r["cost_low"]),
            cost_high=float(r["cost_high"]),
            hours_low=float(r["hours_low"]),
            hours_high=float(r["hours_high"]),
            notes=r["notes"],
        )
        for r in rows
    }


def _load_costs() -> dict[str, ServiceEstimate]:
    try:
        from src.configuration.mongo_db_connection import MongoDBClient
        client = MongoDBClient()
        docs = list(client.database[_REPAIR_COSTS_COLLECTION].find({}, {"_id": 0}))
        if docs:
            logging.info(f"Loaded {len(docs)} repair cost rows from MongoDB.")
            return _rows_to_dict(docs)
        logging.warning("repair_costs collection is empty, using hardcoded fallback.")
    except Exception as e:
        logging.warning(f"MongoDB repair_costs fetch failed: {e} — using hardcoded fallback.")
    return _rows_to_dict(_HARDCODED_COSTS)


def pick_service(inputs: dict[str, Any], prob: float) -> ServiceEstimate:
    """Pick the most relevant service row given the inputs and prediction score."""
    costs = _load_costs()
    brake = (inputs.get("Brake_Condition") or "").lower()
    tire = (inputs.get("Tire_Condition") or "").lower()
    battery = (inputs.get("Battery_Status") or "").lower()
    odo = float(inputs.get("Odometer_Reading") or 0)
    age = int(inputs.get("Vehicle_Age") or 0)
    issues = int(inputs.get("Reported_Issues") or 0)

    flags = []
    if "worn" in brake: flags.append("brake_worn")
    if "worn" in tire: flags.append("tire_worn")
    if "weak" in battery: flags.append("battery_weak")
    if odo > 150000: flags.append("high_mileage_trans")
    if age > 10: flags.append("engine_age")

    # If multiple things are wrong, recommend a full inspection.
    if len(flags) >= 3 and "multiple_flags" in costs:
        return costs["multiple_flags"]
    if "brake_worn" in flags and "engine_age" in flags and "brake_worn_severe" in costs:
        return costs["brake_worn_severe"]
    if "battery_weak" in flags and issues >= 3 and "battery_weak_severe" in costs:
        return costs["battery_weak_severe"]
    for f in flags:
        if f in costs:
            return costs[f]
    # No clear red flag but model still predicts maintenance → routine service.
    if prob >= 0.5 and "high_mileage_routine" in costs:
        return costs["high_mileage_routine"]
    return costs.get("default", next(iter(costs.values())))


# Weight table: each rule maps an input condition to a contribution score.
# Scores are tuned so a typical "heavy maintenance" profile sums near +1.0
# and a healthy profile sits near 0.
def feature_impacts(inputs: dict[str, Any]) -> list[FeatureImpact]:
    out: list[FeatureImpact] = []

    brake = (inputs.get("Brake_Condition") or "").strip()
    tire = (inputs.get("Tire_Condition") or "").strip()
    battery = (inputs.get("Battery_Status") or "").strip()
    odo = float(inputs.get("Odometer_Reading") or 0)
    age = int(inputs.get("Vehicle_Age") or 0)
    issues = int(inputs.get("Reported_Issues") or 0)
    accidents = int(inputs.get("Accident_History") or 0)
    mpg = float(inputs.get("Fuel_Efficiency") or 0)

    brake_score = {"Worn Out": 0.38, "Good": -0.05, "New": -0.18}.get(brake, 0.0)
    out.append(FeatureImpact("Brake Condition", brake, brake_score))

    tire_score = {"Worn Out": 0.24, "Good": -0.04, "New": -0.12}.get(tire, 0.0)
    out.append(FeatureImpact("Tire Condition", tire, tire_score))

    bat_score = {"Weak": 0.22, "Good": -0.03, "Strong": -0.15}.get(battery, 0.0)
    out.append(FeatureImpact("Battery Status", battery, bat_score))

    # Odometer: normalize around 100k km as "typical mid-life"
    odo_score = max(-0.2, min(0.32, (odo - 100_000) / 200_000))
    out.append(FeatureImpact("Odometer Reading", f"{int(odo):,} km", odo_score))

    age_score = max(-0.1, min(0.26, (age - 6) * 0.035))
    out.append(FeatureImpact("Vehicle Age", f"{age} yrs", age_score))

    issues_score = min(0.28, issues * 0.08)
    out.append(FeatureImpact("Reported Issues", issues, issues_score))

    acc_score = min(0.18, accidents * 0.07)
    out.append(FeatureImpact("Accident History", accidents, acc_score))

    # Lower mpg → worse; use a typical 14 km/l reference
    if mpg > 0:
        mpg_score = max(-0.12, min(0.14, (14 - mpg) * 0.02))
        out.append(FeatureImpact("Fuel Efficiency", f"{mpg:.1f} km/l", mpg_score))

    # Sort so the most influential (by magnitude) comes first
    out.sort(key=lambda f: abs(f.contribution), reverse=True)
    return out
