"""Rule-based cost estimation and feature-impact explanation.

Kept deliberately simple and deterministic so the demo is reliable:
  - Cost/hours are looked up from data/repair_costs.csv (industry ballpark ranges).
  - Feature impact is a heuristic "contribution score" computed from the user's
    inputs; it mirrors what a SHAP plot would show but has no external deps and
    can't fail at demo time.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any

_COSTS_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "repair_costs.csv")


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


def _load_costs() -> dict[str, ServiceEstimate]:
    rows: dict[str, ServiceEstimate] = {}
    with open(_COSTS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["trigger"]] = ServiceEstimate(
                service=r["service"],
                cost_low=float(r["cost_low"]),
                cost_high=float(r["cost_high"]),
                hours_low=float(r["hours_low"]),
                hours_high=float(r["hours_high"]),
                notes=r["notes"],
            )
    return rows


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
