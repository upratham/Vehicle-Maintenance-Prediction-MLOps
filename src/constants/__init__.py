import os

from dotenv import load_dotenv

load_dotenv()

# strip surrounding quotes that GH Actions sometimes leaves in secrets
for _k in list(os.environ):
    _v = os.environ[_k]
    if len(_v) >= 2 and _v[0] == _v[-1] and _v[0] in ("'", '"'):
        os.environ[_k] = _v[1:-1]

PLOT_DIR = "plots"

DATABASE_NAME = os.getenv("DB_USERNAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"
TARGET_COLUMN = "NEED_MAINTENANCE"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

AWS_ACCESS_KEY_ID_ENV_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY_ENV_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = "us-east-1"

# Data ingestion
DATA_DIR = "data"
DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

# Data validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# Data transformation
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
PREPROCESSOR_OBJ_DIR: str = "preprocessor_obj"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

ORDINAL_FEATURES = {
    "Maintenance_History": ["Poor", "Average", "Good"],
    "Tire_Condition":      ["Worn Out", "Good", "New"],
    "Brake_Condition":     ["Worn Out", "Good", "New"],
    "Battery_Status":      ["Weak", "Good", "Strong"],
}
DROP_FEATURES = [
    "Maintenance_History", "Mileage", "Last_Service_Date", "Warranty_Expiry_Date",
    "Insurance_Premium", "Service_History", "Owner_Type",
]
NOMINAL_FEATURES = ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"]

# Model trainer
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
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

# Model evaluation / S3 registry
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "vehicle-maintenance-prediction-model"
MODEL_PUSHER_S3_KEY = "model-registry"

# Pipeline artifact archiving (stored in the same bucket as the model)
ARTIFACT_S3_PREFIX = "pipeline-artifacts"
MAX_ARTIFACT_RUNS = 3

# Log retention
LOG_S3_PREFIX = "pipeline-logs"
MAX_LOG_FILES = 3

APP_HOST = "0.0.0.0"
APP_PORT = 5000
