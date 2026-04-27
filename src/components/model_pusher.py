import hashlib
import json
import os
from datetime import datetime, timezone

from src.cloud_storage.aws_storage import SimpleStorageService
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
    if not baselines:
        try:
            from src.artifact_store import load_baselines
            doc = load_baselines()
            if doc:
                baselines = doc.get("models", [])
        except Exception as exc:
            logging.warning(f"baselines MongoDB fallback failed: {exc}")

    # Normalize metrics to the keys the Ops UI expects.
    # The evaluation artifact only carries f1 scalars; the full metrics live
    # on the winner baseline entry produced by model_trainer._train_baselines.
    expected_keys = {"f1", "precision", "recall", "roc_auc", "accuracy"}
    if not expected_keys.issubset(metrics.keys()):
        winner = next((b for b in baselines if b.get("winner")), None)
        if winner:
            metrics = {k: winner.get(k) for k in expected_keys}

    _FAMILY_LABEL = {
        "ensemble-bagging-hpo": "Random Forest (HPO)",
        "ensemble-bagging":     "Random Forest",
        "kernel-svm-hpo":       "SVM (HPO)",
        "kernel-svm":           "SVM",
        "ensemble-boosting":    "XGBoost",
        "linear":               "Logistic Regression",
    }
    _FAMILY_FRAMEWORK = {
        "ensemble-bagging-hpo": "scikit-learn",
        "ensemble-bagging":     "scikit-learn",
        "kernel-svm-hpo":       "scikit-learn",
        "kernel-svm":           "scikit-learn",
        "ensemble-boosting":    "xgboost",
        "linear":               "scikit-learn",
    }
    raw_family = (winner.get("family", "") if winner else "") or ""
    model_family = _FAMILY_LABEL.get(raw_family, raw_family or "Unknown")
    framework = _FAMILY_FRAMEWORK.get(raw_family, "scikit-learn")

    version = datetime.now(timezone.utc).strftime("v%Y.%m.%d-%H%M")
    manifest = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sha256": _sha256(trained_model_path) if os.path.exists(trained_model_path) else None,
        "s3_uri": f"s3://{bucket_name}/{s3_key}",
        "model_family": model_family,
        "framework": framework,
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
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.proj1_estimator = Proj1Estimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Uploading new model to S3 bucket")
        self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
        model_pusher_artifact = ModelPusherArtifact(
            bucket_name=self.model_pusher_config.bucket_name,
            s3_model_path=self.model_pusher_config.s3_model_key_path,
            profile_name=self.model_pusher_config.training_pipeline_config.profile_name,
        )

        try:
            metrics = getattr(self.model_evaluation_artifact, "metrics", None) or {}
            manifest = write_model_registry(
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
            from src.artifact_store import save_model_registry
            save_model_registry(
                profile_name=self.model_pusher_config.training_pipeline_config.profile_name,
                manifest=manifest,
            )

            # Upload manifest + baselines to stable S3 key co-located with the model
            # so the Ops console can always find them even when artifact runs are pruned
            stable_prefix = self.model_pusher_config.s3_model_key_path.rsplit("/", 1)[0]
            bucket = self.model_pusher_config.bucket_name
            self.s3.s3_resource.Object(bucket, f"{stable_prefix}/model_registry.json").put(
                Body=json.dumps(manifest, indent=2).encode()
            )
            logging.info(f"Uploaded manifest -> s3://{bucket}/{stable_prefix}/model_registry.json")

            baselines_path = os.path.join(
                self.model_pusher_config.training_pipeline_config.artifact_dir, "baselines.json"
            )
            if os.path.exists(baselines_path):
                with open(baselines_path, "rb") as bf:
                    self.s3.s3_resource.Object(bucket, f"{stable_prefix}/baselines.json").put(Body=bf.read())
                logging.info(f"Uploaded baselines -> s3://{bucket}/{stable_prefix}/baselines.json")

            # Upload training_distribution.json so drift PSI works in production
            from src.pipeline.training_pipeline import TrainingPipelineConfig
            artifact_dir = self.model_pusher_config.training_pipeline_config.artifact_dir
            profile_name = self.model_pusher_config.training_pipeline_config.profile_name
            td_path = os.path.join(artifact_dir, "training_distribution.json")
            if not os.path.exists(td_path):
                # walk the profile artifact dir for the file
                profile_base = os.path.join("artifact", profile_name)
                if os.path.isdir(profile_base):
                    for run in sorted(os.listdir(profile_base), reverse=True):
                        candidate = os.path.join(profile_base, run, "training_distribution.json")
                        if os.path.exists(candidate):
                            td_path = candidate
                            break
            if os.path.exists(td_path):
                with open(td_path, "rb") as tf:
                    self.s3.s3_resource.Object(bucket, f"{stable_prefix}/training_distribution.json").put(Body=tf.read())
                logging.info(f"Uploaded training_distribution -> s3://{bucket}/{stable_prefix}/training_distribution.json")

        except Exception as reg_err:
            logging.warning(f"model_registry write skipped: {reg_err}")

        return model_pusher_artifact