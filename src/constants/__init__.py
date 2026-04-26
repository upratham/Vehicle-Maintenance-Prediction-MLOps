import os
from datetime import date
from dotenv import load_dotenv
import pandas as pd
load_dotenv()  # never override vars already injected by docker-compose / CI
# For MongoDB connection
REFERENCE_DATE = pd.Timestamp("2026-03-07")
PLOT_DIR = "plots"

DATABASE_NAME = os.getenv("DB_USERNAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MONGODB_URL_KEY = os.getenv("CONNECTION_URL")

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "NEED_MAINTENANCE"
CURRENT_YEAR = date.today().year


FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY_ENV_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = "us-east-1"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_DIR= "data"
DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25
VEHICLE_MAINTENANCE_DATA_FILE_PATH: str = "vehicle_maintenance_data.csv"

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
PREPROCESSOR_OBJ_DIR: str = "preprocessor_obj"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
ORDINAL_FEATURES={
            "Maintenance_History": ["Poor", "Average", "Good"],
            "Tire_Condition":      ["Worn Out", "Good", "New"],
            "Brake_Condition":     ["Worn Out", "Good", "New"],
            "Battery_Status":      ["Weak", "Good", "Strong"],
        }
DROP_FEATURES=["Maintenance_History","Mileage", "Last_Service_Date", "Warranty_Expiry_Date","Insurance_Premium", "Service_History", "Owner_Type"]
NOMINAL_FEATURES = ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"]

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
PARAM = {
        "random_state": 42,
        "hpo_trials": 20,
        "hpo_cv_folds": 3,
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "adam",          
        "activation": "relu",        
        "num_layers": 2,
        "units_layer_1": 128,
        "units_layer_2": 64,
        "units_layer_3": 32,
        "dropout": 0.3,
        "l2_reg": 1e-4,
        "batch_norm": True,
        "threshold": 0.5,
    }

"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "vehicle-maintenance-prediction-model"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 5000
