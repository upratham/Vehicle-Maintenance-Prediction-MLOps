import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_model_registry(
    trained_model_path: str,
    bucket_name: str,
    s3_key: str,
    registry_path: str,
    metrics: dict,
    params: dict | None = None,
    baselines_path: str | None = None,
) -> dict:
    """Write a small JSON manifest describing the freshly-pushed model."""
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    baselines = []
    if baselines_path and os.path.exists(baselines_path):
        try:
            with open(baselines_path) as f:
                baselines = json.load(f).get("models", [])
        except Exception as exc:
            logging.warning(f"baselines.json unreadable: {exc}")

    # Normalize metrics to the keys the Ops UI expects.
    # The evaluation artifact only carries f1 scalars; the full metrics live
    # on the winner baseline entry produced by model_trainer._train_baselines.
    expected_keys = {"f1", "precision", "recall", "roc_auc", "accuracy"}
    if not expected_keys.issubset(metrics.keys()):
        winner = next((b for b in baselines if b.get("winner")), None)
        if winner:
            metrics = {k: winner.get(k) for k in expected_keys}

    version = datetime.now(timezone.utc).strftime("v%Y.%m.%d-%H%M")
    manifest = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sha256": _sha256(trained_model_path) if os.path.exists(trained_model_path) else None,
        "s3_uri": f"s3://{bucket_name}/{s3_key}",
        "model_family": "Keras ANN (binary classifier)",
        "framework": "TensorFlow 2.x",
        "params": params or {},
        "metrics": metrics,
        "baselines": baselines,
        "promoted": True,
    }
    with open(registry_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Wrote model registry manifest -> {registry_path} ({version})")
    return manifest


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.proj1_estimator = Proj1Estimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=model_pusher_config.s3_model_key_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            print("------------------------------------------------------------------------------------------------")

            
            logging.info("Uploading new model to S3 bucket....")
            self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
                profile_name=self.model_pusher_config.training_pipeline_config.profile_name,
            )

            try:
                metrics = getattr(self.model_evaluation_artifact, "metrics", None) or {}
                write_model_registry(
                    trained_model_path=self.model_evaluation_artifact.trained_model_path,
                    bucket_name=self.model_pusher_config.bucket_name,
                    s3_key=self.model_pusher_config.s3_model_key_path,
                    registry_path=self.model_pusher_config.model_registry_file_path,
                    metrics=metrics,
                    baselines_path=os.path.join(
                        self.model_pusher_config.training_pipeline_config.artifact_dir,
                        "baselines.json",
                    ),
                )
            except Exception as reg_err:
                logging.warning(f"model_registry write skipped: {reg_err}")

            logging.info("Uploaded new model to S3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e