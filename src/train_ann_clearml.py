import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import pandas as pd
from clearml import Task, Logger
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from data_preprocessing import main as preprocess_main


def build_model(input_dim, params):
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


def main():
    task = Task.init(
        project_name="605-Vehicle_Maintainance-project",
        task_name="ANN Binary Classification Base",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )

    logger = Logger.current_logger()

    params = {
        "random_state": 42,
        "epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "adam",          # adam, rmsprop, sgd
        "activation": "relu",         # relu, elu, tanh
        "num_layers": 2,
        "units_layer_1": 128,
        "units_layer_2": 64,
        "units_layer_3": 32,
        "dropout": 0.3,
        "l2_reg": 1e-4,
        "batch_norm": True,
        "threshold": 0.5,
    }

    params = task.connect(params)

    np.random.seed(params["random_state"])
    tf.random.set_seed(params["random_state"])

    
    X_train, X_val, y_train, y_val = preprocess_main()

    X_train = np.asarray(X_train).astype("float32")
    X_val = np.asarray(X_val).astype("float32")
    y_train = np.asarray(y_train).astype("float32")
    y_val = np.asarray(y_val).astype("float32")

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)

    model = build_model(X_train.shape[1], params)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    y_prob = model.predict(X_val, verbose=0).ravel()
    y_pred = (y_prob >= params["threshold"]).astype(int)

    val_auc = roc_auc_score(y_val, y_prob)
    val_f1 = f1_score(y_val, y_pred)
    val_precision = precision_score(y_val, y_pred, zero_division=0)
    val_recall = recall_score(y_val, y_pred, zero_division=0)

    print("\nValidation Metrics")
    print(f"AUC      : {val_auc:.5f}")
    print(f"F1       : {val_f1:.5f}")
    print(f"Precision: {val_precision:.5f}")
    print(f"Recall   : {val_recall:.5f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    logger.report_scalar("validation", "auc", iteration=0, value=float(val_auc))
    logger.report_scalar("validation", "f1", iteration=0, value=float(val_f1))
    logger.report_scalar("validation", "precision", iteration=0, value=float(val_precision))
    logger.report_scalar("validation", "recall", iteration=0, value=float(val_recall))

    # log loss curves
    for i, loss in enumerate(history.history.get("loss", [])):
        logger.report_scalar("train", "loss", iteration=i, value=float(loss))

    for i, val_loss in enumerate(history.history.get("val_loss", [])):
        logger.report_scalar("validation", "loss", iteration=i, value=float(val_loss))

    for i, auc in enumerate(history.history.get("auc", [])):
        logger.report_scalar("train", "auc", iteration=i, value=float(auc))

    for i, val_auc_epoch in enumerate(history.history.get("val_auc", [])):
        logger.report_scalar("validation", "auc_epoch", iteration=i, value=float(val_auc_epoch))

    os.makedirs("outputs", exist_ok=True)
    model_path = os.path.join("outputs", "best_ann_model.keras")
    model.save(model_path)
    task.upload_artifact("best_model", artifact_object=model_path)

    print(f"\nClearML Task ID: {task.id}")
    ID=task.id
    task.close()

    return ID


if __name__ == "__main__":
    main()