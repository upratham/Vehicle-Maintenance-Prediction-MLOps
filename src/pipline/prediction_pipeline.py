import sys
import numpy as np
import pandas as pd
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.constants import *
from src.utils.main_utils import load_object
from pandas import DataFrame
import glob
import os


def get_latest_preprocessor_path():
    """Find the most recently saved preprocessor from artifact directory."""
    file_path = os.path.join(PREPROCESSOR_OBJ_DIR, PREPROCSSING_OBJECT_FILE_NAME)
    print(f"Looking for preprocessor at: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No preprocessor found matching: {file_path}")
    else:
        logging.info(f"Using preprocessor: {file_path}")
    return file_path


class VehicleData:
    def __init__(self,
                 Mileage,
                 Reported_Issues,
                 Vehicle_Age,
                 Engine_Size,
                 Odometer_Reading,
                 Insurance_Premium,
                 Service_History,
                 Accident_History,
                 Fuel_Efficiency,
                 Maintenance_History,
                 Tire_Condition,
                 Brake_Condition,
                 Battery_Status,
                 Vehicle_Model,
                 Fuel_Type,
                 Transmission_Type,
                 Owner_Type,
                 Last_Service_Date,
                 Warranty_Expiry_Date
                 ):
        """
        VehicleData constructor
        Input: all 19 raw features required for prediction
        """
        try:
            # Numerical
            self.Mileage = float(Mileage)
            self.Reported_Issues = int(Reported_Issues)
            self.Vehicle_Age = int(Vehicle_Age)
            self.Engine_Size = float(Engine_Size)
            self.Odometer_Reading = float(Odometer_Reading)
            self.Insurance_Premium = float(Insurance_Premium)
            self.Service_History = int(Service_History)
            self.Accident_History = int(Accident_History)
            self.Fuel_Efficiency = float(Fuel_Efficiency)

            # Ordinal categorical
            self.Maintenance_History = Maintenance_History
            self.Tire_Condition = Tire_Condition
            self.Brake_Condition = Brake_Condition
            self.Battery_Status = Battery_Status

            # Nominal categorical
            self.Vehicle_Model = Vehicle_Model
            self.Fuel_Type = Fuel_Type
            self.Transmission_Type = Transmission_Type
            self.Owner_Type = Owner_Type

            # Date fields
            self.Last_Service_Date = Last_Service_Date
            self.Warranty_Expiry_Date = Warranty_Expiry_Date

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self) -> DataFrame:
        """
        Returns a preprocessed DataFrame ready for model prediction.
        Applies the same transformations used during training:
        - Date → days conversion
        - Returns raw DataFrame (preprocessor is applied inside the model/pipeline)
        """
        try:
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            df = DataFrame(vehicle_input_dict)

            # Convert date columns to days (same as training)
            for col in ['Last_Service_Date', 'Warranty_Expiry_Date']:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col + '_days'] = (REFERENCE_DATE - df[col]).dt.days
                df.drop(columns=[col], inplace=True)

            logging.info(f"Prediction input columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_data_as_dict(self):
        """
        Returns raw input as a dictionary (before date conversion).
        """
        logging.info("Entered get_vehicle_data_as_dict method of VehicleData class")
        try:
            input_data = {
                # Numerical
                "Mileage":            [self.Mileage],
                "Reported_Issues":    [self.Reported_Issues],
                "Vehicle_Age":        [self.Vehicle_Age],
                "Engine_Size":        [self.Engine_Size],
                "Odometer_Reading":   [self.Odometer_Reading],
                "Insurance_Premium":  [self.Insurance_Premium],
                "Service_History":    [self.Service_History],
                "Accident_History":   [self.Accident_History],
                "Fuel_Efficiency":    [self.Fuel_Efficiency],

                # Ordinal categorical
                "Maintenance_History": [self.Maintenance_History],
                "Tire_Condition":      [self.Tire_Condition],
                "Brake_Condition":     [self.Brake_Condition],
                "Battery_Status":      [self.Battery_Status],

                # Nominal categorical
                "Vehicle_Model":       [self.Vehicle_Model],
                "Fuel_Type":           [self.Fuel_Type],
                "Transmission_Type":   [self.Transmission_Type],
                "Owner_Type":          [self.Owner_Type],

                # Date fields (raw — converted in get_vehicle_input_data_frame)
                "Last_Service_Date":      [self.Last_Service_Date],
                "Warranty_Expiry_Date":   [self.Warranty_Expiry_Date],
            }

            logging.info("Created vehicle data dict")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e


class VehicleDataClassifier:
    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        Loads preprocessor locally and model from S3.
        Applies preprocessing then runs prediction.
        Returns: numpy array of predictions
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")

            # Step 1 — Load and apply the saved preprocessor
            preprocessor_path = get_latest_preprocessor_path()
            preprocessor = load_object(file_path=preprocessor_path)
            X_transformed = preprocessor.transform(dataframe)
            logging.info(f"Preprocessor applied. Transformed shape: {X_transformed.shape}")

            # Step 2 — Load model from S3 and predict
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(X_transformed)
            return result

        except Exception as e:
            raise MyException(e, sys)
        
if __name__ == "__main__":
    get_latest_preprocessor_path()