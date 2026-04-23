import os
from dataclasses import dataclass, field
from datetime import datetime

from src.constants import *


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    profile_name: str = "vehicle_maintenance"
    pipeline_name: str = PIPELINE_NAME
    timestamp: str = TIMESTAMP
    artifact_dir: str = ""

    def __post_init__(self):
        if not self.artifact_dir:
            self.artifact_dir = os.path.join(ARTIFACT_DIR, self.profile_name, self.timestamp)


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    database_name: str = DATABASE_NAME or "605_Project_Data"
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    data_ingestion_dir: str = ""
    feature_store_file_path: str = ""
    training_file_path: str = ""
    testing_file_path: str = ""

    def __post_init__(self):
        self.data_ingestion_dir = os.path.join(self.training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
        )
        self.training_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
        self.testing_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)


@dataclass
class DataValidationConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    schema_file_path: str = SCHEMA_FILE_PATH
    data_validation_dir: str = ""
    validation_report_file_path: str = ""

    def __post_init__(self):
        self.data_validation_dir = os.path.join(self.training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
        self.validation_report_file_path = os.path.join(self.data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)


@dataclass
class DataTransformationConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    schema_file_path: str = SCHEMA_FILE_PATH
    target_column: str = TARGET_COLUMN
    drop_features: list = field(default_factory=lambda: DROP_FEATURES)
    nominal_features: list = field(default_factory=lambda: NOMINAL_FEATURES)
    ordinal_features: dict = field(default_factory=lambda: ORDINAL_FEATURES)
    training_distribution_path: str = ""
    data_transformation_dir: str = ""
    transformed_object_file_path: str = ""
    transformed_train_file_path: str = ""
    transformed_test_file_path: str = ""

    def __post_init__(self):
        self.data_transformation_dir = os.path.join(
            self.training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_object_file_path = os.path.join(
            PREPROCESSOR_OBJ_DIR,
            self.training_pipeline_config.profile_name,
            PREPROCSSING_OBJECT_FILE_NAME,
        )
        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME.replace("csv", "npy"),
        )
        if not self.training_distribution_path:
            self.training_distribution_path = os.path.join(
                self.training_pipeline_config.artifact_dir,
                "training_distribution.json",
            )


@dataclass
class ModelTrainerConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    model_type: str = "ann"
    target_column: str = TARGET_COLUMN
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
    params: dict = field(default_factory=lambda: PARAM)
    model_trainer_dir: str = ""
    trained_model_file_path: str = ""
    baselines_file_path: str = ""

    def __post_init__(self):
        self.model_trainer_dir = os.path.join(self.training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_TRAINED_MODEL_DIR,
            MODEL_FILE_NAME,
        )
        self.baselines_file_path = os.path.join(self.training_pipeline_config.artifact_dir, "baselines.json")


@dataclass
class ModelEvaluationConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = ""

    def __post_init__(self):
        if not self.s3_model_key_path:
            self.s3_model_key_path = os.path.join(
                MODEL_PUSHER_S3_KEY,
                self.training_pipeline_config.profile_name,
                MODEL_FILE_NAME,
            ).replace("\\", "/")


@dataclass
class ModelPusherConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = ""
    model_registry_file_path: str = ""

    def __post_init__(self):
        if not self.s3_model_key_path:
            self.s3_model_key_path = os.path.join(
                MODEL_PUSHER_S3_KEY,
                self.training_pipeline_config.profile_name,
                MODEL_FILE_NAME,
            ).replace("\\", "/")
        if not self.model_registry_file_path:
            self.model_registry_file_path = os.path.join(
                self.training_pipeline_config.artifact_dir,
                "model_registry.json",
            )


@dataclass
class VehiclePredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME