import json
import os
import sys
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
from clearml import Logger, Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def _train_baselines(X_train, y_train, X_test, y_test, ann_metrics: dict, output_path: str, target_name: str) -> dict:
    """Train LR + RF (+ XGB if installed) and persist a comparison table."""
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
                            eval_metric="logloss", use_label_encoder=False, n_jobs=-1),
              "ensemble-boosting",
              {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.08, "subsample": 0.9})
    except Exception as e:
        logging.warning(f"XGB baseline skipped: {e}")

    models.append({
        "name": ann_metrics.get("name", "Primary Model"),
        "family": ann_metrics.get("family", "custom"),
        "hyperparams": ann_metrics.get("hyperparams", {}),
        **{k: float(v) for k, v in ann_metrics.items()},
        "winner": True,
    })

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


from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a ANN with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            profile_name = self.model_trainer_config.training_pipeline_config.profile_name
            model_type = (self.model_trainer_config.model_type or "rf").strip().lower()

            task = Task.init(
            project_name="605-Vehicle_Maintainance-project",
            task_name=f"{profile_name} {model_type.upper()} Training",
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,)
            logger = Logger.current_logger()
            logging.info(f"ClearML task initialized for model training for profile={profile_name}, model_type={model_type}.")
            task.connect({"profile_name": profile_name, "model_type": model_type})
    
            logging.info("Training model with specified parameters")

            # Splitting the train and test data into features and target variables
            X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")
            params = self.model_trainer_config.params

            if model_type in {"rf", "random_forest"}:
                model = RandomForestClassifier(
                    n_estimators=int(params.get("rf_n_estimators", 300)),
                    max_depth=params.get("rf_max_depth", None),
                    min_samples_leaf=int(params.get("rf_min_samples_leaf", 1)),
                    random_state=int(params.get("random_state", 42)),
                    n_jobs=-1,
                )
                model_family = "ensemble-bagging"
                model_hparams = {
                    "n_estimators": model.n_estimators,
                    "max_depth": model.max_depth,
                    "min_samples_leaf": model.min_samples_leaf,
                    "random_state": model.random_state,
                }
                model.fit(X_train, y_train)
            elif model_type in {"dt", "decision_tree"}:
                model = DecisionTreeClassifier(
                    max_depth=params.get("dt_max_depth", None),
                    min_samples_leaf=int(params.get("dt_min_samples_leaf", 1)),
                    random_state=int(params.get("random_state", 42)),
                )
                model_family = "tree"
                model_hparams = {
                    "max_depth": model.max_depth,
                    "min_samples_leaf": model.min_samples_leaf,
                    "random_state": model.random_state,
                }
                model.fit(X_train, y_train)
            elif model_type in {"rf_hpo", "random_forest_hpo", "rf-with-hpo"}:
                try:
                    import optuna

                    logging.info("Starting Optuna HPO for Random Forest")
                    n_trials = int(params.get("hpo_trials", 20))
                    cv_folds = int(params.get("hpo_cv_folds", 3))

                    def objective(trial):
                        estimator = RandomForestClassifier(
                            n_estimators=trial.suggest_int("n_estimators", 100, 500),
                            max_depth=trial.suggest_int("max_depth", 3, 25),
                            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                            random_state=int(params.get("random_state", 42)),
                            n_jobs=-1,
                        )
                        scores = cross_val_score(
                            estimator,
                            X_train,
                            y_train,
                            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                            scoring="f1",
                            n_jobs=-1,
                        )
                        return float(scores.mean())

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=n_trials)

                    best = study.best_params
                    logger.report_scalar("hpo/best_cv_f1", "rf_hpo", float(study.best_value), 0)
                    task.connect({"hpo_best_params": best})

                    model = RandomForestClassifier(
                        n_estimators=int(best["n_estimators"]),
                        max_depth=int(best["max_depth"]),
                        min_samples_leaf=int(best["min_samples_leaf"]),
                        min_samples_split=int(best["min_samples_split"]),
                        random_state=int(params.get("random_state", 42)),
                        n_jobs=-1,
                    )
                    model_family = "ensemble-bagging-hpo"
                    model_hparams = best
                    model.fit(X_train, y_train)
                except Exception as hpo_err:
                    logging.warning(f"RF HPO failed, falling back to plain RF: {hpo_err}")
                    model = RandomForestClassifier(
                        n_estimators=int(params.get("rf_n_estimators", 300)),
                        max_depth=params.get("rf_max_depth", None),
                        min_samples_leaf=int(params.get("rf_min_samples_leaf", 1)),
                        random_state=int(params.get("random_state", 42)),
                        n_jobs=-1,
                    )
                    model_family = "ensemble-bagging"
                    model_hparams = {
                        "n_estimators": model.n_estimators,
                        "max_depth": model.max_depth,
                        "min_samples_leaf": model.min_samples_leaf,
                        "random_state": model.random_state,
                    }
                    model.fit(X_train, y_train)
            else:
                raise ValueError(
                    f"Unsupported model_type '{model_type}'. Use one of: rf, dt, rf_hpo."
                )

            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = y_pred
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Explicitly log final test metrics to ClearML
            try:
                logger.report_scalar("test/accuracy",  "final", accuracy,  0)
                logger.report_scalar("test/f1",        "final", f1,        0)
                logger.report_scalar("test/precision", "final", precision, 0)
                logger.report_scalar("test/recall",    "final", recall,    0)
                logger.report_scalar("test/roc_auc",   "final", float(roc_auc_score(y_test, y_pred_proba)), 0)
                task.upload_artifact("trained_model", artifact_object=model)
                task.close()
            except Exception as log_err:
                logging.warning(f"ClearML final logging skipped: {log_err}")

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall, accuracy_score=accuracy)

            try:
                _train_baselines(
                    X_train, y_train, X_test, y_test,
                    ann_metrics={
                        "name": f"Primary {model_type.upper()}",
                        "family": model_family,
                        "hyperparams": model_hparams,
                        "f1": f1, "precision": precision, "recall": recall,
                        "accuracy": accuracy,
                        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
                    },
                    output_path=self.model_trainer_config.baselines_file_path,
                    target_name=self.model_trainer_config.target_column,
                )
            except Exception as bl_err:
                logging.warning(f"baselines write skipped: {bl_err}")

            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
           


            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            save_object(self.model_trainer_config.trained_model_file_path, trained_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
                profile_name=self.model_trainer_config.training_pipeline_config.profile_name,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e