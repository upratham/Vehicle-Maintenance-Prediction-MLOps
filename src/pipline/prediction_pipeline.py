import os
import sys
from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame

from src.constants import (
    MODEL_BUCKET_NAME,
    MODEL_FILE_NAME,
    MODEL_PUSHER_S3_KEY,
    PREPROCESSOR_OBJ_DIR,
    PREPROCSSING_OBJECT_FILE_NAME,
)
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Base profile classifier
# ---------------------------------------------------------------------------

class BaseProfileClassifier(ABC):
    """
    Loads the profile-specific preprocessor from disk and the model from S3,
    then runs inference on a pre-built DataFrame.
    Subclasses only need to declare PROFILE_NAME and implement label().
    """

    PROFILE_NAME: str = ""

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        try:
            logging.info(f"[{self.PROFILE_NAME}] Loading preprocessor.")
            preprocessor = load_object(_preprocessor_path(self.PROFILE_NAME))
            X = preprocessor.transform(dataframe)
            logging.info(f"[{self.PROFILE_NAME}] Transformed shape: {X.shape}")

            logging.info(f"[{self.PROFILE_NAME}] Loading model from S3.")
            estimator = Proj1Estimator(
                bucket_name=MODEL_BUCKET_NAME,
                model_path=_s3_model_key(self.PROFILE_NAME),
            )
            return estimator.predict(X)
        except Exception as e:
            raise MyException(e, sys) from e

    @abstractmethod
    def label(self, value: int) -> str:
        """Human-readable label for a binary prediction (0 or 1)."""


# ---------------------------------------------------------------------------
# Profile-specific classifiers
# ---------------------------------------------------------------------------

class VehicleMaintenanceClassifier(BaseProfileClassifier):
    PROFILE_NAME = "vehicle_maintenance"

    def label(self, value: int) -> str:
        return "Maintenance Required" if value == 1 else "No Maintenance Needed"


class HyundaiCarsClassifier(BaseProfileClassifier):
    PROFILE_NAME = "cars_hyundai"

    def label(self, value: int) -> str:
        return "Anomaly Detected" if value == 1 else "Normal"


class EngineDataClassifier(BaseProfileClassifier):
    PROFILE_NAME = "engine_data"

    def label(self, value: int) -> str:
        return "Engine Issue Detected" if value == 1 else "Engine Healthy"


# ---------------------------------------------------------------------------
# Input data classes
# ---------------------------------------------------------------------------

class VehicleMaintenanceData:
    """
    Input features for the vehicle_maintenance model.
    Columns match those present in the raw dataset (pre-preprocessor).
    The preprocessor applies MI filtering / OHE / scaling internally.
    """

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
        try:
            self.Reported_Issues   = int(Reported_Issues)
            self.Vehicle_Age       = int(Vehicle_Age)
            self.Engine_Size       = float(Engine_Size)
            self.Odometer_Reading  = float(Odometer_Reading)
            self.Accident_History  = int(Accident_History)
            self.Fuel_Efficiency   = float(Fuel_Efficiency)
            self.Tire_Condition    = Tire_Condition
            self.Brake_Condition   = Brake_Condition
            self.Battery_Status    = Battery_Status
            self.Vehicle_Model     = Vehicle_Model
            self.Fuel_Type         = Fuel_Type
            self.Transmission_Type = Transmission_Type
        except Exception as e:
            raise MyException(e, sys) from e

    def get_dataframe(self) -> DataFrame:
        try:
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
        except Exception as e:
            raise MyException(e, sys) from e

    # backward-compat alias used by the old app.py form route
    def get_vehicle_input_data_frame(self) -> DataFrame:
        return self.get_dataframe()


class HyundaiCarsData:
    """
    Input features for the cars_hyundai anomaly-detection model.

      engine_temperature  -> "Engine Temperature (°C)"  float
      brake_pad_thickness -> "Brake Pad Thickness (mm)" float
      tire_pressure       -> "Tire Pressure (PSI)"      float
      maintenance_type    -> "Maintenance Type"          str   (e.g. "Preventive", "Corrective")
    """

    def __init__(
        self,
        engine_temperature: float,
        brake_pad_thickness: float,
        tire_pressure: float,
        maintenance_type: str,
    ):
        try:
            self.engine_temperature  = float(engine_temperature)
            self.brake_pad_thickness = float(brake_pad_thickness)
            self.tire_pressure       = float(tire_pressure)
            self.maintenance_type    = str(maintenance_type)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_dataframe(self) -> DataFrame:
        try:
            return DataFrame({
                "Engine Temperature (°C)": [self.engine_temperature],
                "Brake Pad Thickness (mm)": [self.brake_pad_thickness],
                "Tire Pressure (PSI)":      [self.tire_pressure],
                "Maintenance Type":          [self.maintenance_type],
            })
        except Exception as e:
            raise MyException(e, sys) from e


class EngineData:
    """
    Input features for the engine_data condition model (all numeric).

      engine_rpm       -> "Engine rpm"
      lub_oil_pressure -> "Lub oil pressure"
      fuel_pressure    -> "Fuel pressure"
      coolant_pressure -> "Coolant pressure"
      lub_oil_temp     -> "lub oil temp"
      coolant_temp     -> "Coolant temp"
    """

    def __init__(
        self,
        engine_rpm: float,
        lub_oil_pressure: float,
        fuel_pressure: float,
        coolant_pressure: float,
        lub_oil_temp: float,
        coolant_temp: float,
    ):
        try:
            self.engine_rpm       = float(engine_rpm)
            self.lub_oil_pressure = float(lub_oil_pressure)
            self.fuel_pressure    = float(fuel_pressure)
            self.coolant_pressure = float(coolant_pressure)
            self.lub_oil_temp     = float(lub_oil_temp)
            self.coolant_temp     = float(coolant_temp)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_dataframe(self) -> DataFrame:
        try:
            return DataFrame({
                "Engine rpm":       [self.engine_rpm],
                "Lub oil pressure": [self.lub_oil_pressure],
                "Fuel pressure":    [self.fuel_pressure],
                "Coolant pressure": [self.coolant_pressure],
                "lub oil temp":     [self.lub_oil_temp],
                "Coolant temp":     [self.coolant_temp],
            })
        except Exception as e:
            raise MyException(e, sys) from e


# ---------------------------------------------------------------------------
# Multi-model orchestrator
# ---------------------------------------------------------------------------

class MultiModelOrchestrator:
    """
    Routes a prediction request to the correct profile-specific classifier.

    Example:
        result = MultiModelOrchestrator().predict("cars_hyundai", df)
        # {"profile": "cars_hyundai", "label": 1, "score": 0.82, "status": "Anomaly Detected"}
    """

    CLASSIFIERS: dict[str, type[BaseProfileClassifier]] = {
        "vehicle_maintenance": VehicleMaintenanceClassifier,
        "cars_hyundai":        HyundaiCarsClassifier,
        "engine_data":         EngineDataClassifier,
    }

    def predict(self, profile_name: str, dataframe: DataFrame) -> dict:
        try:
            classifier_class = self.CLASSIFIERS.get(profile_name)
            if classifier_class is None:
                raise ValueError(
                    f"Unknown profile '{profile_name}'. "
                    f"Valid: {list(self.CLASSIFIERS)}"
                )
            classifier = classifier_class()
            raw = classifier.predict(dataframe)[0]
            score = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
            label = 1 if score >= 0.5 else 0
            return {
                "profile": profile_name,
                "label":   label,
                "score":   round(score, 4),
                "status":  classifier.label(label),
            }
        except Exception as e:
            raise MyException(e, sys) from e


# ---------------------------------------------------------------------------
# Backward-compatible aliases  (the legacy app.py form route imports these)
# ---------------------------------------------------------------------------

VehicleData = VehicleMaintenanceData


class VehicleDataClassifier:
    """
    Thin wrapper kept for backward compatibility with the legacy HTML form route.
    Delegates to VehicleMaintenanceClassifier internally.
    """

    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig()):
        self._config = prediction_pipeline_config

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        try:
            preprocessor = load_object(_preprocessor_path("vehicle_maintenance"))
            X = preprocessor.transform(dataframe)
            estimator = Proj1Estimator(
                bucket_name=self._config.model_bucket_name,
                model_path=_s3_model_key("vehicle_maintenance"),
            )
            return estimator.predict(X)
        except Exception as e:
            raise MyException(e, sys) from e
