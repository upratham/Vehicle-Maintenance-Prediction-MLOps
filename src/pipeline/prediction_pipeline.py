import os
import pickle

import numpy as np
from pandas import DataFrame

from src.constants import (
    MODEL_BUCKET_NAME,
    MODEL_FILE_NAME,
    MODEL_PUSHER_S3_KEY,
    PREPROCESSOR_OBJ_DIR,
    PREPROCSSING_OBJECT_FILE_NAME,
)
from src.logger import logging
from src.utils.main_utils import load_object


PROFILE_LABELS = {
    "vehicle_maintenance": ("Maintenance Required", "No Maintenance Needed"),
    "cars_hyundai": ("Anomaly Detected", "Normal"),
    "engine_data": ("Engine Issue Detected", "Engine Healthy"),
}


def load_preprocessor(profile_name):
    path = os.path.join(PREPROCESSOR_OBJ_DIR, profile_name, PREPROCSSING_OBJECT_FILE_NAME)
    if os.path.exists(path):
        return load_object(path)
    raise FileNotFoundError(
        f"no preprocessor for {profile_name}. run `python demo.py` to train models first."
    )


def load_model(profile_name):
    base = os.path.join("artifact", profile_name)
    if os.path.isdir(base):
        runs = sorted(
            (d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))),
            reverse=True,
        )
        for run in runs:
            p = os.path.join(base, run, "model_trainer", "trained_model", MODEL_FILE_NAME)
            if os.path.exists(p):
                return load_object(p)

    try:
        from src.cloud_storage.aws_storage import SimpleStorageService
        s3 = SimpleStorageService()
        key = f"{MODEL_PUSHER_S3_KEY}/{profile_name}/{MODEL_FILE_NAME}"
        obj = s3.s3_resource.Object(MODEL_BUCKET_NAME, key).get()
        return pickle.loads(obj["Body"].read())
    except Exception as e:
        logging.warning(f"model not found in s3 either: {e}")
        raise FileNotFoundError(
            f"no model for {profile_name}. run `python demo.py` to train models first."
        )


class ProfileClassifier:
    def __init__(self, profile_name):
        if profile_name not in PROFILE_LABELS:
            raise ValueError(f"unknown profile {profile_name}")
        self.profile_name = profile_name

    def predict(self, dataframe):
        preprocessor = load_preprocessor(self.profile_name)
        X = preprocessor.transform(dataframe)
        model = load_model(self.profile_name)
        return model.predict(X)

    def label(self, value):
        on, off = PROFILE_LABELS[self.profile_name]
        return on if value == 1 else off


class VehicleMaintenanceData:
    def __init__(
        self,
        Reported_Issues,
        Vehicle_Age,
        Engine_Size,
        Odometer_Reading,
        Accident_History,
        Fuel_Efficiency,
        Tire_Condition,
        Brake_Condition,
        Battery_Status,
        Vehicle_Model,
        Fuel_Type,
        Transmission_Type,
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

    def get_dataframe(self):
        return DataFrame({
            "Reported_Issues": [self.Reported_Issues],
            "Vehicle_Age": [self.Vehicle_Age],
            "Engine_Size": [self.Engine_Size],
            "Odometer_Reading": [self.Odometer_Reading],
            "Accident_History": [self.Accident_History],
            "Fuel_Efficiency": [self.Fuel_Efficiency],
            "Tire_Condition": [self.Tire_Condition],
            "Brake_Condition": [self.Brake_Condition],
            "Battery_Status": [self.Battery_Status],
            "Vehicle_Model": [self.Vehicle_Model],
            "Fuel_Type": [self.Fuel_Type],
            "Transmission_Type": [self.Transmission_Type],
        })


class HyundaiCarsData:
    def __init__(self, engine_temperature, brake_pad_thickness, tire_pressure, maintenance_type):
        self.engine_temperature = float(engine_temperature)
        self.brake_pad_thickness = float(brake_pad_thickness)
        self.tire_pressure = float(tire_pressure)
        self.maintenance_type = str(maintenance_type)

    def get_dataframe(self):
        return DataFrame({
            "Engine Temperature (°C)": [self.engine_temperature],
            "Brake Pad Thickness (mm)": [self.brake_pad_thickness],
            "Tire Pressure (PSI)": [self.tire_pressure],
            "Maintenance Type": [self.maintenance_type],
        })


class EngineData:
    def __init__(self, engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp):
        self.engine_rpm = float(engine_rpm)
        self.lub_oil_pressure = float(lub_oil_pressure)
        self.fuel_pressure = float(fuel_pressure)
        self.coolant_pressure = float(coolant_pressure)
        self.lub_oil_temp = float(lub_oil_temp)
        self.coolant_temp = float(coolant_temp)

    def get_dataframe(self):
        return DataFrame({
            "Engine rpm": [self.engine_rpm],
            "Lub oil pressure": [self.lub_oil_pressure],
            "Fuel pressure": [self.fuel_pressure],
            "Coolant pressure": [self.coolant_pressure],
            "lub oil temp": [self.lub_oil_temp],
            "Coolant temp": [self.coolant_temp],
        })


def predict_profile(profile_name, dataframe):
    classifier = ProfileClassifier(profile_name)
    raw = classifier.predict(dataframe)[0]
    score = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
    label = 1 if score >= 0.5 else 0
    return {
        "profile": profile_name,
        "label": label,
        "score": round(score, 4),
        "status": classifier.label(label),
    }
