import os

from src.cloud_storage.MongoDB import get_collection_name_from_csv, push_csv_to_mongodb
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.constants import (
    DATA_DIR, DATABASE_NAME, DROP_FEATURES, NOMINAL_FEATURES, ORDINAL_FEATURES,
    MODEL_BUCKET_NAME, ARTIFACT_S3_PREFIX, MAX_ARTIFACT_RUNS,
    LOG_S3_PREFIX, MAX_LOG_FILES,
)
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
            "paths": {
                "schema": os.path.join("config", "schema.yaml"),
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
        paths_cfg = profile_cfg.get("paths", {})
        preprocess_cfg = profile_cfg.get("preprocessing", {})

        schema_path = paths_cfg.get("schema", os.path.join("config", "schema.yaml"))
        collection_name = dataset_cfg.get("collection", profile_name)
        target_column = dataset_cfg.get("target_column", "Need_Maintenance")
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
            target_column=target_column,
        )
        self.model_evaluation_config = ModelEvaluationConfig(
            training_pipeline_config=self.training_pipeline_config,
        )
        self.model_pusher_config = ModelPusherConfig(
            training_pipeline_config=self.training_pipeline_config,
        )

    def _sync_artifact_to_s3(self, profile_name: str, artifact_dir: str, timestamp: str) -> None:
        """Upload artifact_dir to S3, prune old S3 runs, and prune old local runs."""
        import shutil
        try:
            from src.cloud_storage.aws_storage import SimpleStorageService
            s3 = SimpleStorageService()
            run_prefix = f"{ARTIFACT_S3_PREFIX}/{profile_name}/{timestamp}"
            s3.upload_directory(artifact_dir, run_prefix, MODEL_BUCKET_NAME)

            # Prune old S3 artifact runs
            profile_prefix = f"{ARTIFACT_S3_PREFIX}/{profile_name}"
            all_s3_runs = s3.list_run_prefixes(MODEL_BUCKET_NAME, profile_prefix)
            stale_s3 = all_s3_runs[:-MAX_ARTIFACT_RUNS]
            for old_prefix in stale_s3:
                logging.info(f"[artifact-prune] Removing S3 run: s3://{MODEL_BUCKET_NAME}/{old_prefix}")
                s3.delete_prefix(MODEL_BUCKET_NAME, old_prefix)
            logging.info(
                f"[artifact-prune] S3 {profile_name}: kept {len(all_s3_runs) - len(stale_s3)}, "
                f"pruned {len(stale_s3)}"
            )
        except Exception as exc:
            logging.warning(f"[artifact-prune] S3 sync skipped for {profile_name}: {exc}")

        # Prune old local artifact runs (independent of S3 success)
        try:
            local_base = os.path.join("artifact", profile_name)
            if os.path.isdir(local_base):
                local_runs = sorted(
                    d for d in os.listdir(local_base)
                    if os.path.isdir(os.path.join(local_base, d))
                )
                for old_run in local_runs[:-MAX_ARTIFACT_RUNS]:
                    shutil.rmtree(os.path.join(local_base, old_run), ignore_errors=True)
                    logging.info(f"[artifact-prune] Removed local run: artifact/{profile_name}/{old_run}")
        except Exception as exc:
            logging.warning(f"[artifact-prune] local prune skipped for {profile_name}: {exc}")

    def _sync_logs_to_s3(self) -> None:
        """Upload the current log file to S3 and prune old S3 log objects to MAX_LOG_FILES."""
        try:
            from src.cloud_storage.aws_storage import SimpleStorageService
            from src.logger import LOG_FILE, log_file_path
            s3 = SimpleStorageService()

            # Upload current log
            s3_key = f"{LOG_S3_PREFIX}/{LOG_FILE}"
            s3.s3_resource.meta.client.upload_file(log_file_path, MODEL_BUCKET_NAME, s3_key)
            logging.info(f"[log-sync] Uploaded log -> s3://{MODEL_BUCKET_NAME}/{s3_key}")

            # Prune old S3 log objects
            all_logs = []
            paginator = s3.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=MODEL_BUCKET_NAME, Prefix=f"{LOG_S3_PREFIX}/"):
                for obj in page.get("Contents", []):
                    all_logs.append(obj)
            all_logs.sort(key=lambda o: o["LastModified"])
            stale_logs = all_logs[:-MAX_LOG_FILES]
            if stale_logs:
                s3.s3_client.delete_objects(
                    Bucket=MODEL_BUCKET_NAME,
                    Delete={"Objects": [{"Key": o["Key"]} for o in stale_logs]},
                )
                logging.info(f"[log-sync] Pruned {len(stale_logs)} old S3 log(s)")
        except Exception as exc:
            logging.warning(f"[log-sync] S3 log sync skipped: {exc}")

    def start_csv_to_mongodb_sync(self, target_collections: list[str] | None = None) -> None:
        """Sync CSV files from DATA_DIR into MongoDB; optionally limit to a subset of collections."""
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
                continue
            result = push_csv_to_mongodb(csv_path=csv_path, db_name=mongo_db_name)
            logging.info(f"CSV sync result for {csv_path}: {result}")

    def start_data_ingestion(self) -> DataIngestionArtifact:
        return DataIngestion(data_ingestion_config=self.data_ingestion_config).initiate_data_ingestion()

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        return DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=self.data_validation_config,
        ).initiate_data_validation()

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        return DataTransformation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_transformation_config=self.data_transformation_config,
            data_validation_artifact=data_validation_artifact,
        ).initiate_data_transformation()

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        return ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=self.model_trainer_config,
        ).initiate_model_trainer()

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelEvaluationArtifact:
        return ModelEvaluation(
            model_eval_config=self.model_evaluation_config,
            data_ingestion_artifact=data_ingestion_artifact,
            model_trainer_artifact=model_trainer_artifact,
            data_transformation_artifact=data_transformation_artifact,
        ).initiate_model_evaluation()

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        return ModelPusher(
            model_evaluation_artifact=model_evaluation_artifact,
            model_pusher_config=self.model_pusher_config,
        ).initiate_model_pusher()

    def run_pipeline(
        self,
        refresh_collections: bool = False,
        profile_names: list[str] | None = None,
    ) -> dict:
        """Run training pipeline for one or more profiles; return per-profile status and metrics."""
        profiles = self._load_profiles()
        selected_profiles = profile_names or list(profiles.keys())
        unknown_profiles = [name for name in selected_profiles if name not in profiles]
        if unknown_profiles:
            raise ValueError(f"Unknown profile(s): {unknown_profiles}")

        if refresh_collections:
            target_collections = [
                profiles[name].get("dataset", {}).get("collection", name)
                for name in selected_profiles
            ]
            logging.info(f"Syncing collections for selected profiles -> {target_collections}")
            self.start_csv_to_mongodb_sync(target_collections=target_collections)

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

            base_result = {
                "trained_model_path": model_trainer_artifact.trained_model_file_path,
                "preprocessor_path": data_transformation_artifact.transformed_object_file_path,
                "metrics": model_evaluation_artifact.metrics or {},
            }

            if not model_evaluation_artifact.is_model_accepted:
                logging.info(f"[{profile_name}] Model not accepted.")
                results[profile_name] = {**base_result, "accepted": False, "pushed": False}
                self._sync_artifact_to_s3(
                    profile_name, self.training_pipeline_config.artifact_dir,
                    self.training_pipeline_config.timestamp,
                )
                continue

            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            results[profile_name] = {
                **base_result,
                "accepted": True,
                "pushed": True,
                "s3_model_path": model_pusher_artifact.s3_model_path,
            }
            self._sync_artifact_to_s3(
                profile_name, self.training_pipeline_config.artifact_dir,
                self.training_pipeline_config.timestamp,
            )
            logging.info(f"==================== [{profile_name}] pipeline complete ====================")

        self._sync_logs_to_s3()
        return results
