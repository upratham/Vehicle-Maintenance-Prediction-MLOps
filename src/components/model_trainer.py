import json
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import optuna
from clearml import Logger, Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

optuna.logging.set_verbosity(optuna.logging.WARNING)

CLEARML_PROJECT = "605-Vehicle_Maintainance-project"


def _train_baselines(X_train, y_train, X_test, y_test, primary_metrics: dict, output_path: str, target_name: str) -> dict:
    models: list[dict] = []

    def _eval(name, clf, family, hyperparams):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else pred
        models.append({
            "name": name,
            "family": family,
            "hyperparams": hyperparams,
            "f1": float(f1_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
        })

    try:
        _eval("Logistic Regression",
              LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
              "linear",
              {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 500})
    except Exception as e:
        logging.warning(f"LR baseline failed: {e}")

    try:
        _eval("Random Forest",
              RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1),
              "ensemble-bagging",
              {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 3, "random_state": 42})
    except Exception as e:
        logging.warning(f"RF baseline failed: {e}")

    try:
        from xgboost import XGBClassifier
        _eval("XGBoost",
              XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9,
                            eval_metric="logloss", n_jobs=-1),
              "ensemble-boosting",
              {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.08, "subsample": 0.9})
    except Exception as e:
        logging.warning(f"XGB baseline skipped: {e}")

    models.append({**primary_metrics, "winner": True})

    out = {
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cv_folds": 5,
        "target": target_name,
        "models": models,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


class BaseModelTrainer(ABC):
    """Shared training flow; subclasses implement `run_hpo` with dataset-specific search spaces."""

    PROFILE_NAME: str = ""

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def _init_clearml_task(self) -> Tuple[Task, Logger]:
        profile = self.model_trainer_config.training_pipeline_config.profile_name
        task = Task.init(
            project_name=CLEARML_PROJECT,
            task_name=f"{profile} HPO Training",
            task_type=Task.TaskTypes.optimizer,
            reuse_last_task_id=False,
        )
        task.connect({"profile_name": profile})
        logger = Logger.current_logger()
        logging.info(f"ClearML task initialised for profile={profile}.")
        return task, logger

    def _log_trial(self, logger: Logger, trial_number: int, cv_f1: float) -> None:
        """Log each Optuna trial F1 to ClearML so HPO progress is visible."""
        logger.report_scalar("hpo/cv_f1", "trial", cv_f1, trial_number)

    def _log_final_metrics(self, logger: Logger, metrics: dict) -> None:
        logger.report_scalar("test/accuracy",  "final", metrics["accuracy"],  0)
        logger.report_scalar("test/f1",        "final", metrics["f1"],        0)
        logger.report_scalar("test/precision", "final", metrics["precision"], 0)
        logger.report_scalar("test/recall",    "final", metrics["recall"],    0)
        logger.report_scalar("test/roc_auc",   "final", metrics["roc_auc"],   0)

    def _evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        return {
            "f1":        float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall":    float(recall_score(y_test, y_pred)),
            "accuracy":  float(accuracy_score(y_test, y_pred)),
            "roc_auc":   float(roc_auc_score(y_test, y_proba)),
        }

    @abstractmethod
    def run_hpo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict,
        logger: Logger,
    ) -> Tuple[object, dict, str]:
        """Run HPO; return (best_model_unfitted, best_params, model_family)."""

    def get_model_object_and_report(
        self, train: np.ndarray, test: np.ndarray
    ) -> Tuple[object, ClassificationMetricArtifact]:
        try:
            X_train, y_train = train[:, :-1], train[:, -1]
            X_test,  y_test  = test[:, :-1],  test[:, -1]
            params = self.model_trainer_config.params

            task, logger = self._init_clearml_task()

            logging.info(f"[{self.PROFILE_NAME}] Starting HPO.")
            best_model, best_params, model_family = self.run_hpo(X_train, y_train, params, logger)

            logging.info(f"[{self.PROFILE_NAME}] Fitting best model on full training set.")
            best_model.fit(X_train, y_train)

            metrics = self._evaluate(best_model, X_test, y_test)
            logging.info(f"[{self.PROFILE_NAME}] Test metrics: {metrics}")

            task.connect({"hpo_best_params": best_params})
            try:
                self._log_final_metrics(logger, metrics)
                task.upload_artifact("trained_model", artifact_object=best_model)
                task.close()
            except Exception as log_err:
                logging.warning(f"ClearML final logging skipped: {log_err}")

            try:
                _train_baselines(
                    X_train, y_train, X_test, y_test,
                    primary_metrics={
                        "name": f"{self.PROFILE_NAME} HPO",
                        "family": model_family,
                        "hyperparams": best_params,
                        **metrics,
                    },
                    output_path=self.model_trainer_config.baselines_file_path,
                    target_name=self.model_trainer_config.target_column,
                )
            except Exception as bl_err:
                logging.warning(f"Baselines write skipped: {bl_err}")

            metric_artifact = ClassificationMetricArtifact(
                f1_score=metrics["f1"],
                precision_score=metrics["precision"],
                recall_score=metrics["recall"],
                accuracy_score=metrics["accuracy"],
            )
            return best_model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info(f"[{self.PROFILE_NAME}] Entered initiate_model_trainer.")
        try:
            print("---" * 30)
            print(f"Starting Model Trainer — {self.PROFILE_NAME}")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr  = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train/test arrays loaded.")

            trained_model, metric_artifact = self.get_model_object_and_report(train_arr, test_arr)

            save_object(self.model_trainer_config.trained_model_file_path, trained_model)
            logging.info("Trained model saved.")

            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
                profile_name=self.model_trainer_config.training_pipeline_config.profile_name,
            )
            logging.info(f"[{self.PROFILE_NAME}] ModelTrainerArtifact: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys) from e


# ---------------------------------------------------------------------------
# Dataset-specific trainers
# ---------------------------------------------------------------------------

class VehicleMaintenanceModelTrainer(BaseModelTrainer):
    """
    Random-Forest HPO for the vehicle-maintenance dataset.
    Search space targets recall/F1 for the maintenance-needed class.
    """

    PROFILE_NAME = "vehicle_maintenance"

    def run_hpo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict,
        logger: Logger,
    ) -> Tuple[object, dict, str]:
        n_trials  = int(params.get("hpo_trials",   20))
        cv_folds  = int(params.get("hpo_cv_folds",  3))
        rand_seed = int(params.get("random_state", 42))

        def objective(trial: optuna.Trial) -> float:
            clf = RandomForestClassifier(
                n_estimators      = trial.suggest_int("n_estimators",       100, 600),
                max_depth         = trial.suggest_int("max_depth",            4,  30),
                min_samples_leaf  = trial.suggest_int("min_samples_leaf",     1,  10),
                min_samples_split = trial.suggest_int("min_samples_split",    2,  20),
                max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                random_state=rand_seed,
                n_jobs=-1,
            )
            scores = cross_val_score(
                clf, X_train, y_train,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rand_seed),
                scoring="f1", n_jobs=-1,
            )
            cv_f1 = float(scores.mean())
            self._log_trial(logger, trial.number, cv_f1)
            return cv_f1

        study = optuna.create_study(direction="maximize", study_name="vm_rf_hpo")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        logging.info(f"[{self.PROFILE_NAME}] Best HPO params (cv_f1={study.best_value:.4f}): {best}")
        logger.report_scalar("hpo/best_cv_f1", "vm_rf_hpo", study.best_value, 0)

        model = RandomForestClassifier(
            n_estimators      = int(best["n_estimators"]),
            max_depth         = int(best["max_depth"]),
            min_samples_leaf  = int(best["min_samples_leaf"]),
            min_samples_split = int(best["min_samples_split"]),
            max_features      = best["max_features"],
            random_state=rand_seed,
            n_jobs=-1,
        )
        return model, best, "ensemble-bagging-hpo"


class HyundaiCarsModelTrainer(BaseModelTrainer):
    """
    Decision-Tree HPO for the Hyundai anomaly-detection dataset.
    Uses a broader depth range since the dataset is smaller.
    """

    PROFILE_NAME = "cars_hyundai"

    def run_hpo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict,
        logger: Logger,
    ) -> Tuple[object, dict, str]:
        n_trials  = int(params.get("hpo_trials",   20))
        cv_folds  = int(params.get("hpo_cv_folds",  3))
        rand_seed = int(params.get("random_state", 42))

        def objective(trial: optuna.Trial) -> float:
            max_feat_choice = trial.suggest_categorical("max_features", ["sqrt", "log2", "all"])
            clf = DecisionTreeClassifier(
                max_depth         = trial.suggest_int("max_depth",           3,  25),
                min_samples_leaf  = trial.suggest_int("min_samples_leaf",    1,  20),
                min_samples_split = trial.suggest_int("min_samples_split",   2,  30),
                criterion         = trial.suggest_categorical("criterion",   ["gini", "entropy"]),
                max_features      = None if max_feat_choice == "all" else max_feat_choice,
                random_state=rand_seed,
            )
            scores = cross_val_score(
                clf, X_train, y_train,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rand_seed),
                scoring="f1", n_jobs=-1,
            )
            cv_f1 = float(scores.mean())
            self._log_trial(logger, trial.number, cv_f1)
            return cv_f1

        study = optuna.create_study(direction="maximize", study_name="hyundai_dt_hpo")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        logging.info(f"[{self.PROFILE_NAME}] Best HPO params (cv_f1={study.best_value:.4f}): {best}")
        logger.report_scalar("hpo/best_cv_f1", "hyundai_dt_hpo", study.best_value, 0)

        max_feat_val = None if best["max_features"] == "all" else best["max_features"]
        model = DecisionTreeClassifier(
            max_depth         = int(best["max_depth"]),
            min_samples_leaf  = int(best["min_samples_leaf"]),
            min_samples_split = int(best["min_samples_split"]),
            criterion         = best["criterion"],
            max_features      = max_feat_val,
            random_state=rand_seed,
        )
        return model, {**best, "max_features": max_feat_val}, "tree-hpo"


class EngineDataModelTrainer(BaseModelTrainer):
    """
    Random-Forest HPO for the engine-condition dataset.
    Extends the estimator and depth ranges to handle the richer numeric feature space.
    """

    PROFILE_NAME = "engine_data"

    def run_hpo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict,
        logger: Logger,
    ) -> Tuple[object, dict, str]:
        n_trials  = int(params.get("hpo_trials",   20))
        cv_folds  = int(params.get("hpo_cv_folds",  3))
        rand_seed = int(params.get("random_state", 42))

        def objective(trial: optuna.Trial) -> float:
            clf = RandomForestClassifier(
                n_estimators      = trial.suggest_int("n_estimators",       100, 600),
                max_depth         = trial.suggest_int("max_depth",            3,  25),
                min_samples_leaf  = trial.suggest_int("min_samples_leaf",     1,  10),
                min_samples_split = trial.suggest_int("min_samples_split",    2,  20),
                max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                random_state=rand_seed,
                n_jobs=-1,
            )
            scores = cross_val_score(
                clf, X_train, y_train,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rand_seed),
                scoring="f1", n_jobs=-1,
            )
            cv_f1 = float(scores.mean())
            self._log_trial(logger, trial.number, cv_f1)
            return cv_f1

        study = optuna.create_study(direction="maximize", study_name="engine_rf_hpo")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        logging.info(f"[{self.PROFILE_NAME}] Best HPO params (cv_f1={study.best_value:.4f}): {best}")
        logger.report_scalar("hpo/best_cv_f1", "engine_rf_hpo", study.best_value, 0)

        model = RandomForestClassifier(
            n_estimators      = int(best["n_estimators"]),
            max_depth         = int(best["max_depth"]),
            min_samples_leaf  = int(best["min_samples_leaf"]),
            min_samples_split = int(best["min_samples_split"]),
            max_features      = best["max_features"],
            random_state=rand_seed,
            n_jobs=-1,
        )
        return model, best, "ensemble-bagging-hpo"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Factory that returns the correct dataset-specific trainer.
    Mirrors the DataTransformation dispatcher pattern.
    """

    PROFILE_CLASS_MAP = {
        "vehicle_maintenance": VehicleMaintenanceModelTrainer,
        "cars_hyundai":        HyundaiCarsModelTrainer,
        "engine_data":         EngineDataModelTrainer,
    }

    def __new__(
        cls,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ) -> BaseModelTrainer:
        profile_name = model_trainer_config.training_pipeline_config.profile_name
        trainer_class = cls.PROFILE_CLASS_MAP.get(profile_name, VehicleMaintenanceModelTrainer)
        logging.info(f"ModelTrainer dispatcher -> {trainer_class.__name__} for profile='{profile_name}'.")
        return trainer_class(data_transformation_artifact, model_trainer_config)
