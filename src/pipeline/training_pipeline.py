import os
import sys

from src.cloud_storage.MongoDB import get_collection_name_from_csv, push_csv_to_mongodb
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.constants import DATA_DIR, DATABASE_NAME, DROP_FEATURES, NOMINAL_FEATURES, ORDINAL_FEATURES
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file

PROFILES_CONFIG_PATH = os.path.join("config", "pipeline_profiles.yaml")
DEFAULT_DB_NAME = "605_Project_Data"


class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self._set_profile_context("vehicle_maintenance", {
            "dataset": {
                "collection": "vehicle_maintenance_data",
                "target_column": "Need_Maintenance",
            },
            "model": {
                "type": "ann",
            },
            "paths": {
                "schema": os.path.join("config", "schema.yaml"),
                "model_config": os.path.join("config", "model.yaml"),
            },
        })

    def _load_profiles(self) -> dict:
        profiles_cfg = read_yaml_file(PROFILES_CONFIG_PATH) or {}
        profiles = profiles_cfg.get("profiles", {})
        if not profiles:
            raise ValueError(
                f"No profiles found in {PROFILES_CONFIG_PATH}. Add at least one profile under 'profiles'."
            )
        return profiles

    def _set_profile_context(self, profile_name: str, profile_cfg: dict) -> None:
        dataset_cfg = profile_cfg.get("dataset", {})
        model_cfg = profile_cfg.get("model", {})
        paths_cfg = profile_cfg.get("paths", {})
        preprocess_cfg = profile_cfg.get("preprocessing", {})

        schema_path = paths_cfg.get("schema", os.path.join("config", "schema.yaml"))
        model_cfg_path = paths_cfg.get("model_config", os.path.join("config", "model.yaml"))
        collection_name = dataset_cfg.get("collection", profile_name)
        target_column = dataset_cfg.get("target_column", "Need_Maintenance")
        model_type = model_cfg.get("type", "ann")
        db_name = DATABASE_NAME or DEFAULT_DB_NAME

        self.training_pipeline_config = TrainingPipelineConfig(profile_name=profile_name)

        self.data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=self.training_pipeline_config,
            collection_name=collection_name,
            database_name=db_name,
        )
        self.data_validation_config = DataValidationConfig(
            training_pipeline_config=self.training_pipeline_config,
            schema_file_path=schema_path,
        )
        self.data_transformation_config = DataTransformationConfig(
            training_pipeline_config=self.training_pipeline_config,
            schema_file_path=schema_path,
            target_column=target_column,
            drop_features=preprocess_cfg.get("drop_features", DROP_FEATURES),
            nominal_features=preprocess_cfg.get("nominal_features", NOMINAL_FEATURES),
            ordinal_features=preprocess_cfg.get("ordinal_features", ORDINAL_FEATURES),
        )
        self.model_trainer_config = ModelTrainerConfig(
            training_pipeline_config=self.training_pipeline_config,
            model_type=model_type,
            target_column=target_column,
            model_config_file_path=model_cfg_path,
        )
        self.model_evaluation_config = ModelEvaluationConfig(
            training_pipeline_config=self.training_pipeline_config,
        )
        self.model_pusher_config = ModelPusherConfig(
            training_pipeline_config=self.training_pipeline_config,
        )

    def start_csv_to_mongodb_sync(self, target_collections: list[str] | None = None) -> None:
        """
        Sync CSV files from DATA_DIR into MongoDB collections.
        If target_collections is provided, only matching CSV-derived collection names are synced.
        """
        try:
            logging.info("Entered the start_csv_to_mongodb_sync method of TrainPipeline class")
            csv_paths = [
                os.path.join(DATA_DIR, file_name)
                for file_name in os.listdir(DATA_DIR)
                if file_name.endswith(".csv")
            ]
            logging.info(f"Found {len(csv_paths)} CSV files in {DATA_DIR} for MongoDB sync")
            mongo_db_name = DATABASE_NAME or DEFAULT_DB_NAME
            target_set = set(target_collections or [])

            for csv_path in csv_paths:
                collection_name = get_collection_name_from_csv(csv_path)
                if target_set and collection_name not in target_set:
                    logging.info(
                        f"Skipping sync for {csv_path}: collection {collection_name} not in selected profiles"
                    )
                    continue
                result = push_csv_to_mongodb(csv_path=csv_path, db_name=mongo_db_name)
                logging.info(f"CSV sync result for {csv_path}: {result}")

            logging.info("Exited the start_csv_to_mongodb_sync method of TrainPipeline class")
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component.
        """
        try:
            logging.info(
                f"[{self.training_pipeline_config.profile_name}] Entered start_data_ingestion"
            )
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(
                f"[{self.training_pipeline_config.profile_name}] Exited start_data_ingestion"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component.
        """
        logging.info(f"[{self.training_pipeline_config.profile_name}] Entered start_data_validation")
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"[{self.training_pipeline_config.profile_name}] Exited start_data_validation")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component.
        """
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise MyException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training.
        """
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise MyException(e, sys) from e

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting model evaluation.
        """
        try:
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact,
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise MyException(e, sys) from e

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing.
        """
        try:
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(
        self,
        refresh_collections: bool = False,
        profile_names: list[str] | None = None,
    ) -> dict:
        """
        Run training pipeline for one or more profiles.

        Parameters:
            refresh_collections: if True, sync selected profile datasets from CSV into MongoDB first
            profile_names: explicit subset of profiles from config/pipeline_profiles.yaml

        Returns:
            dict keyed by profile name with pipeline status and metrics.
        """
        try:
            profiles = self._load_profiles()
            selected_profiles = profile_names or list(profiles.keys())
            unknown_profiles = [name for name in selected_profiles if name not in profiles]
            if unknown_profiles:
                raise ValueError(f"Unknown profile(s): {unknown_profiles}")

            if refresh_collections:
                target_collections = [profiles[name].get("dataset", {}).get("collection", name) for name in selected_profiles]
                logging.info(
                    f"refresh_collections=True: syncing collections for selected profiles -> {target_collections}"
                )
                self.start_csv_to_mongodb_sync(target_collections=target_collections)
            else:
                logging.info("refresh_collections=False: skipping CSV to MongoDB sync")

            results: dict = {}

            for profile_name in selected_profiles:
                profile_cfg = profiles[profile_name]
                logging.info(f"==================== [{profile_name}] pipeline start ====================")
                self._set_profile_context(profile_name, profile_cfg)

                data_ingestion_artifact = self.start_data_ingestion()
                data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
                data_transformation_artifact = self.start_data_transformation(
                    data_ingestion_artifact=data_ingestion_artifact,
                    data_validation_artifact=data_validation_artifact,
                )
                model_trainer_artifact = self.start_model_trainer(
                    data_transformation_artifact=data_transformation_artifact
                )
                model_evaluation_artifact = self.start_model_evaluation(
                    data_ingestion_artifact=data_ingestion_artifact,
                    model_trainer_artifact=model_trainer_artifact,
                    data_transformation_artifact=data_transformation_artifact,
                )

                if not model_evaluation_artifact.is_model_accepted:
                    logging.info(f"[{profile_name}] Model not accepted.")
                    results[profile_name] = {
                        "accepted": False,
                        "pushed": False,
                        "trained_model_path": model_trainer_artifact.trained_model_file_path,
                        "preprocessor_path": data_transformation_artifact.transformed_object_file_path,
                        "metrics": model_evaluation_artifact.metrics or {},
                    }
                    continue

                model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
                results[profile_name] = {
                    "accepted": True,
                    "pushed": True,
                    "trained_model_path": model_trainer_artifact.trained_model_file_path,
                    "preprocessor_path": data_transformation_artifact.transformed_object_file_path,
                    "s3_model_path": model_pusher_artifact.s3_model_path,
                    "metrics": model_evaluation_artifact.metrics or {},
                }

                logging.info(f"==================== [{profile_name}] pipeline complete ====================")

            return results
        except Exception as e:
            raise MyException(e, sys) from e
