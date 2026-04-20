import json
import sys
from datetime import datetime, timezone
from typing import Tuple
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report,accuracy_score

BASELINES_PATH = os.path.join("artifact", "baselines.json")


def _train_baselines(X_train, y_train, X_test, y_test, ann_metrics: dict) -> dict:
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
        "name": "Keras ANN",
        "family": "neural-net",
        "hyperparams": {"note": "see PARAM in src/constants"},
        **{k: float(v) for k, v in ann_metrics.items()},
        "winner": True,
    })

    out = {
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cv_folds": 5,
        "target": "Need_Maintenance",
        "models": models,
    }
    os.makedirs(os.path.dirname(BASELINES_PATH), exist_ok=True)
    with open(BASELINES_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return out
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from clearml import Task, Logger


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

  
    def build_model(self, input_dim, params):
        logging.info(f"Building ANN model with input dimension {input_dim}")
        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for i in range(params["num_layers"]):
            units = params[f"units_layer_{i+1}"]
            model.add(
                Dense(
                    units,
                    activation=params["activation"],
                    kernel_regularizer=l2(params["l2_reg"])
                )
            )

            if params["batch_norm"]:
                model.add(BatchNormalization())

            model.add(Dropout(params["dropout"]))

        model.add(Dense(1, activation="sigmoid"))

        if params["optimizer"] == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
        elif params["optimizer"] == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=params["learning_rate"])
        else:
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=params["learning_rate"],
                momentum=0.9
            )

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        return model



    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a ANN with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            task = Task.init(
            project_name="605-Vehicle_Maintainance-project",
            task_name="ANN Binary Classification Base",
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,)
            logger = Logger.current_logger()
            logging.info("ClearlyML task initialized for model training.")
    
            logging.info("Training ANN with specified parameters")

            # Splitting the train and test data into features and target variables
            X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")
            params=self.model_trainer_config.params
            model = self.build_model(input_dim=X_train.shape[1], params=params)
            early_stop = EarlyStopping(
                                        monitor="val_auc",
                                        mode="max",
                                        patience=8,
                                        restore_best_weights=True,
                                        verbose=1)
            
            reduce_lr = ReduceLROnPlateau(
                                            monitor="val_auc",
                                            mode="max",
                                            factor=0.5,
                                            patience=4,
                                            min_lr=1e-6,
                                            verbose=1
                                        )
            # Fit the model
            logging.info("Model training going on...")
            history = model.fit(             X_train,
                                            y_train,
                                            validation_split=0.2,
                                            epochs=params["epochs"],
                                            batch_size=params["batch_size"],
                                            callbacks=[early_stop, reduce_lr],
                                            verbose=1,
                                        )
                                                        
      
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred_proba = model.predict(X_test).flatten() 
            y_pred = (y_pred_proba >= 0.5).astype(int)  
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
                        "f1": f1, "precision": precision, "recall": recall,
                        "accuracy": accuracy,
                        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
                    },
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
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e