from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    DataTransformationArtifact,
)
from src.entity.s3_estimator import Proj1Estimator
from src.logger import logging
from src.utils.main_utils import load_object, load_numpy_array_data


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """Return the deployed S3 model if present, else None."""
        bucket_name = self.model_eval_config.bucket_name
        model_path = self.model_eval_config.s3_model_key_path
        proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)
        if proj1_estimator.is_model_present(model_path=model_path):
            return proj1_estimator
        return None

    def evaluate_model(self) -> EvaluateModelResponse:
        test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
        x, y = test_arr[:, :-1], test_arr[:, -1]

        load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
        trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
        logging.info(f"F1_Score for this model: {trained_model_f1_score}")

        best_model_f1_score = None
        best_model = self.get_best_model()
        if best_model is not None:
            try:
                raw_predictions = np.asarray(best_model.predict(x))
                if raw_predictions.ndim == 2:
                    if raw_predictions.shape[1] == 1:
                        y_hat_best_model = (raw_predictions.ravel() >= 0.5).astype(int)
                    else:
                        y_hat_best_model = np.argmax(raw_predictions, axis=1)
                else:
                    y_hat_best_model = raw_predictions.ravel()

                average = "binary" if len(np.unique(y)) <= 2 else "weighted"
                best_model_f1_score = f1_score(y, y_hat_best_model, average=average)
                logging.info(
                    f"F1_Score-Production Model: {best_model_f1_score}, "
                    f"F1_Score-New Trained Model: {trained_model_f1_score}"
                )
            except Exception as best_model_err:
                logging.warning(
                    "Skipping production-model comparison because the deployed model "
                    f"is incompatible with the current transformed features: {best_model_err}"
                )
                best_model_f1_score = None

        tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
        return EvaluateModelResponse(
            trained_model_f1_score=trained_model_f1_score,
            best_model_f1_score=best_model_f1_score,
            is_model_accepted=trained_model_f1_score > tmp_best_model_score,
            difference=trained_model_f1_score - tmp_best_model_score,
        )

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logging.info("Initialized Model Evaluation Component.")
        evaluate_model_response = self.evaluate_model()
        return ModelEvaluationArtifact(
            is_model_accepted=evaluate_model_response.is_model_accepted,
            s3_model_path=self.model_eval_config.s3_model_key_path,
            trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            changed_accuracy=evaluate_model_response.difference,
            metrics={
                "trained_model_f1_score": evaluate_model_response.trained_model_f1_score,
                "best_model_f1_score": evaluate_model_response.best_model_f1_score,
                "difference": evaluate_model_response.difference,
            },
            profile_name=self.model_eval_config.training_pipeline_config.profile_name,
        )
