import os

import numpy as np
from pandas import DataFrame

from src.constants import (
    MODEL_BUCKET_NAME,
    MODEL_FILE_NAME,
    MODEL_PUSHER_S3_KEY,
    PREPROCESSOR_OBJ_DIR,
    PREPROCSSING_OBJECT_FILE_NAME,
)
from src.entity.s3_estimator import Proj1Estimator
from src.utils.main_utils import load_object


def _preprocessor_path(profile_name: str) -> str:
    path = os.path.join(PREPROCESSOR_OBJ_DIR, profile_name, PREPROCSSING_OBJECT_FILE_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No preprocessor for profile '{profile_name}' at: {path}. "
            "Run the training pipeline first."
        )
    return path


def _s3_model_key(profile_name: str) -> str:
    return f"{MODEL_PUSHER_S3_KEY}/{profile_name}/{MODEL_FILE_NAME}".replace("\\", "/")


PROFILE_LABELS: dict[str, tuple[str, str]] = {
    # profile_name : (label_when_1, label_when_0)
    "vehicle_maintenance": ("Maintenance Required", "No Maintenance Needed"),
    "cars_hyundai":        ("Anomaly Detected",     "Normal"),
    "engine_data":         ("Engine Issue Detected", "Engine Healthy"),
}


class ProfileClassifier:
    """Config-driven classifier: loads the profile's preprocessor + S3 model and predicts."""

    def __init__(self, profile_name: str):
        if profile_name not in PROFILE_LABELS:
            raise ValueError(
                f"Unknown profile '{profile_name}'. Valid: {list(PROFILE_LABELS)}"
            )
        self.profile_name = profile_name

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        preprocessor = load_object(_preprocessor_path(self.profile_name))
        X = preprocessor.transform(dataframe)
        estimator = Proj1Estimator(
            bucket_name=MODEL_BUCKET_NAME,
            model_path=_s3_model_key(self.profile_name),
        )
        return estimator.predict(X)

    def label(self, value: int) -> str:
        on, off = PROFILE_LABELS[self.profile_name]
        return on if value == 1 else off


class VehicleMaintenanceData:
    """Input features for the vehicle_maintenance model (raw, pre-preprocessor)."""

    def __init__(
        self,
        Reported_Issues: int,
        Vehicle_Age: int,
        Engine_Size: float,
        Odometer_Reading: float,
        Accident_History: int,
        Fuel_Efficiency: float,
        Tire_Condition: str,
        Brake_Condition: str,
        Battery_Status: str,
        Vehicle_Model: str,
        Fuel_Type: str,
        Transmission_Type: str,
    ):
        self.Reported_Issues = int(Reported_Issues)
        self.Vehicle_Age = int(Vehicle_Age)
        self.Engine_Size = float(Engine_Size)
        self.Odometer_Reading = float(Odometer_Reading)
        self.Accident_History = int(Accident_History)
        self.Fuel_Efficiency = float(Fuel_Efficiency)
        self.Tire_Condition = Tire_Condition
        self.Brake_Condition = Brake_Condition
        self.Battery_Status = Battery_Status
        self.Vehicle_Model = Vehicle_Model
        self.Fuel_Type = Fuel_Type
        self.Transmission_Type = Transmission_Type

    def get_dataframe(self) -> DataFrame:
        return DataFrame({
            "Reported_Issues":   [self.Reported_Issues],
            "Vehicle_Age":       [self.Vehicle_Age],
            "Engine_Size":       [self.Engine_Size],
            "Odometer_Reading":  [self.Odometer_Reading],
            "Accident_History":  [self.Accident_History],
            "Fuel_Efficiency":   [self.Fuel_Efficiency],
            "Tire_Condition":    [self.Tire_Condition],
            "Brake_Condition":   [self.Brake_Condition],
            "Battery_Status":    [self.Battery_Status],
            "Vehicle_Model":     [self.Vehicle_Model],
            "Fuel_Type":         [self.Fuel_Type],
            "Transmission_Type": [self.Transmission_Type],
        })

    def get_vehicle_input_data_frame(self) -> DataFrame:
        """Backward-compat alias used by the legacy form route."""
        return self.get_dataframe()


class HyundaiCarsData:
    """Input features for the cars_hyundai anomaly-detection model."""

    def __init__(
        self,
        engine_temperature: float,
        brake_pad_thickness: float,
        tire_pressure: float,
        maintenance_type: str,
    ):
        self.engine_temperature = float(engine_temperature)
        self.brake_pad_thickness = float(brake_pad_thickness)
        self.tire_pressure = float(tire_pressure)
        self.maintenance_type = str(maintenance_type)

    def get_dataframe(self) -> DataFrame:
        return DataFrame({
            "Engine Temperature (°C)":   [self.engine_temperature],
            "Brake Pad Thickness (mm)":  [self.brake_pad_thickness],
            "Tire Pressure (PSI)":       [self.tire_pressure],
            "Maintenance Type":          [self.maintenance_type],
        })


class EngineData:
    """Input features for the engine_data condition model (all numeric)."""

    def __init__(
        self,
        engine_rpm: float,
        lub_oil_pressure: float,
        fuel_pressure: float,
        coolant_pressure: float,
        lub_oil_temp: float,
        coolant_temp: float,
    ):
        self.engine_rpm = float(engine_rpm)
        self.lub_oil_pressure = float(lub_oil_pressure)
        self.fuel_pressure = float(fuel_pressure)
        self.coolant_pressure = float(coolant_pressure)
        self.lub_oil_temp = float(lub_oil_temp)
        self.coolant_temp = float(coolant_temp)

    def get_dataframe(self) -> DataFrame:
        return DataFrame({
            "Engine rpm":       [self.engine_rpm],
            "Lub oil pressure": [self.lub_oil_pressure],
            "Fuel pressure":    [self.fuel_pressure],
            "Coolant pressure": [self.coolant_pressure],
            "lub oil temp":     [self.lub_oil_temp],
            "Coolant temp":     [self.coolant_temp],
        })


class MultiModelOrchestrator:
    """Route a prediction request to the correct profile-specific classifier."""

    def predict(self, profile_name: str, dataframe: DataFrame) -> dict:
        classifier = ProfileClassifier(profile_name)
        raw = classifier.predict(dataframe)[0]
        score = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
        label = 1 if score >= 0.5 else 0
        return {
            "profile": profile_name,
            "label":   label,
            "score":   round(score, 4),
            "status":  classifier.label(label),
        }


# Backward-compat aliases used by the legacy app.py form route.
VehicleData = VehicleMaintenanceData


class VehicleDataClassifier:
    """Thin wrapper kept for backward compatibility with the legacy HTML form route."""

    def __init__(self, prediction_pipeline_config=None):
        self._classifier = ProfileClassifier("vehicle_maintenance")

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        return self._classifier.predict(dataframe)
