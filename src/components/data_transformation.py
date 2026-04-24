import json
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, RobustScaler

from src.constants import PLOT_DIR
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, save_numpy_array_data, save_object

logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def _write_training_distribution(raw_df: pd.DataFrame, schema: dict, output_path: str) -> None:
    """Persist a snapshot of training feature distributions for later drift checks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    target = schema.get("target_column")
    numerical_cols = [c for c in schema.get("numerical_columns", []) if c != target and c in raw_df.columns]
    categorical_cols = [c for c in schema.get("categorical_columns", []) if c in raw_df.columns]

    numerical = {}
    for col in numerical_cols:
        series = pd.to_numeric(raw_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        hist, edges = np.histogram(series.values, bins=10)
        numerical[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std() or 0.0),
            "bins": [float(edge) for edge in edges.tolist()],
            "counts": [int(count) for count in hist.tolist()],
        }

    categorical = {}
    for col in categorical_cols:
        counts = raw_df[col].astype(str).value_counts().to_dict()
        categorical[col] = {str(key): int(value) for key, value in counts.items()}

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows": int(len(raw_df)),
        "numerical": numerical,
        "categorical": categorical,
    }
    with open(output_path, "w") as file_obj:
        json.dump(snapshot, file_obj, indent=2)
    logging.info(f"Wrote training distribution snapshot -> {output_path}")


class BaseDataTransformation:
    """Shared transformation flow with dataset-specific extension points."""

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=self.data_transformation_config.schema_file_path)
            self.profile_name = getattr(
                self.data_transformation_config.training_pipeline_config,
                "profile_name",
                "",
            )
            self.target_column = self._resolve_target_column(self.data_transformation_config.target_column)
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def _normalize_col_name(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def _is_target_column(self, column_name: str) -> bool:
        return self._normalize_col_name(column_name) == self._normalize_col_name(self.target_column)

    def _resolve_target_column(self, configured_target: str) -> str:
        configured = configured_target or self._schema_config.get("target_column", "")
        schema_target = self._schema_config.get("target_column", "")
        candidates = [configured, schema_target]

        raw_df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path, nrows=1)
        available = list(raw_df.columns)
        normalized_map = {self._normalize_col_name(col): col for col in available}

        for candidate in candidates:
            if candidate and candidate in available:
                return candidate
            normalized = self._normalize_col_name(candidate)
            if normalized in normalized_map:
                resolved = normalized_map[normalized]
                logging.warning(
                    f"Configured target '{candidate}' not found exactly. Using matched column '{resolved}'."
                )
                return resolved

        if available:
            fallback = available[-1]
            logging.warning(
                f"Target column not found in schema/profile. Falling back to last column: {fallback}."
            )
            return fallback
        raise ValueError("Unable to resolve target column: dataset has no columns.")

    def _get_schema_numerical_columns(self, df: pd.DataFrame) -> list[str]:
        return [
            col
            for col in self._schema_config.get("numerical_columns", [])
            if col in df.columns and not self._is_target_column(col)
        ]

    def _get_schema_categorical_columns(self, df: pd.DataFrame) -> list[str]:
        return [
            col
            for col in self._schema_config.get("categorical_columns", [])
            if col in df.columns and not self._is_target_column(col)
        ]

    def handle_duplicates_and_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Handling duplicates and missing values")
        df = df.drop_duplicates()
        df = df.fillna(0)
        return df.reset_index(drop=True)

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def split_features_and_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        logging.info("Splitting features and target variable")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y

    @staticmethod
    def _should_stratify(y: pd.Series) -> bool:
        if y.nunique(dropna=False) <= 1:
            return False
        counts = y.value_counts(dropna=False)
        return bool(not counts.empty and counts.min() >= 2)

    def split_train_test(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        stratify = y if self._should_stratify(y) else None
        if stratify is None:
            logging.info("Skipping stratified split because target distribution is not compatible.")
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    def encode_target(self, y_train: pd.Series, y_test: pd.Series) -> tuple[pd.Series, pd.Series]:
        return y_train, y_test

    def post_split_train_processing(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return X_train, X_test

    def build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        raise NotImplementedError("Subclasses must implement build_preprocessor().")

    def should_apply_smote(self) -> bool:
        return False

    def apply_smote_if_needed(self, X_train: np.ndarray, y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        y_array = np.array(y_train)
        if not self.should_apply_smote():
            logging.info("SMOTE skipped for this dataset profile.")
            return X_train, y_array

        class_counts = pd.Series(y_train).value_counts(dropna=False)
        if len(class_counts) < 2 or class_counts.min() < 2:
            logging.warning(
                "SMOTE skipped because target class distribution is not suitable: "
                f"{class_counts.to_dict()}"
            )
            return X_train, y_array

        logging.info("Applying SMOTE for handling imbalanced dataset.")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, np.array(y_train_resampled)

    def _save_outputs(
        self,
        preprocessor: ColumnTransformer,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: pd.Series,
    ) -> DataTransformationArtifact:
        train_arr = np.c_[X_train, y_train]
        test_arr = np.c_[X_test, np.array(y_test)]

        save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
        save_numpy_array_data(
            self.data_transformation_config.transformed_train_file_path,
            array=train_arr,
        )
        save_numpy_array_data(
            self.data_transformation_config.transformed_test_file_path,
            array=test_arr,
        )
        logging.info("Saved transformed files and preprocessor object.")

        return DataTransformationArtifact(
            transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
            transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            profile_name=self.profile_name,
        )

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"[{self.profile_name}] Data transformation started.")
            if not self.data_validation_artifact.validation_status:
                raise ValueError(self.data_validation_artifact.message)

            df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)
            logging.info(f"[{self.profile_name}] Loaded feature store with shape {df.shape}.")

            try:
                _write_training_distribution(
                    raw_df=df,
                    schema=self._schema_config,
                    output_path=self.data_transformation_config.training_distribution_path,
                )
            except Exception as snap_err:
                logging.warning(f"Training distribution snapshot skipped: {snap_err}")

            df = self.handle_duplicates_and_missing_values(df)
            df = self.prepare_dataframe(df)
            X, y = self.split_features_and_target(df)
            X_train, X_test, y_train, y_test = self.split_train_test(X, y)
            y_train, y_test = self.encode_target(y_train=y_train, y_test=y_test)
            X_train, X_test = self.post_split_train_processing(X_train=X_train, X_test=X_test)

            preprocessor = self.build_preprocessor(X_train)
            preprocessor.fit(X_train)
            X_train_arr = preprocessor.transform(X_train)
            X_test_arr = preprocessor.transform(X_test)
            logging.info(
                f"[{self.profile_name}] Preprocessing complete. "
                f"Train shape={X_train_arr.shape}, Test shape={X_test_arr.shape}"
            )

            X_train_arr, y_train_arr = self.apply_smote_if_needed(X_train_arr, y_train)
            artifact = self._save_outputs(
                preprocessor=preprocessor,
                X_train=X_train_arr,
                y_train=y_train_arr,
                X_test=X_test_arr,
                y_test=y_test,
            )
            logging.info(f"[{self.profile_name}] Data transformation completed successfully.")
            return artifact
        except Exception as e:
            raise MyException(e, sys) from e


class VehicleMaintenanceDataTransformation(BaseDataTransformation):
    """Vehicle-maintenance dataset transformation with richer feature selection."""

    def drop_not_important_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Dropping configured vehicle-maintenance features")
        features_to_drop = [
            col for col in self.data_transformation_config.drop_features if col in df.columns
        ]
        if features_to_drop:
            df = df.drop(columns=features_to_drop)
        return df

    def filter_low_information_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Encoding categorical variables temporarily for MI-based filtering")
        df_encoded = df.copy()
        categorical_cols = self._get_schema_categorical_columns(df)
        label_encoder = LabelEncoder()

        for col in categorical_cols:
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))

        spearman_corr = df_encoded.corr(method="spearman")
        os.makedirs(PLOT_DIR, exist_ok=True)

        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(
            spearman_corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            annot_kws={"size": 7},
        )
        plt.title("Spearman Rank-Correlation Heat-map", fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "spearman_heatmap.png"), dpi=300)
        plt.close()

        X_tmp = df_encoded.drop(columns=[self.target_column])
        y_tmp = df_encoded[self.target_column]
        mi_scores = mutual_info_classif(X_tmp, y_tmp, discrete_features="auto", random_state=42)
        mi_series = pd.Series(mi_scores, index=X_tmp.columns).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        mi_series.plot(kind="bar", color="steelblue")
        plt.title(f"Mutual Information Scores vs. {self.target_column}")
        plt.ylabel("MI Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "mi_scores.png"), dpi=300)
        plt.close()

        zero_mi_features = mi_series[mi_series == 0].index.tolist()
        if zero_mi_features:
            logging.info(f"Dropping zero-MI features: {zero_mi_features}")
            df = df.drop(columns=zero_mi_features)

        logging.info(f"Surviving columns after MI filter ({len(df.columns)}): {df.columns.tolist()}")
        return df

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.drop_not_important_features(df)
        return self.filter_low_information_features(df)

    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Capping outliers in vehicle-maintenance numerical columns")
        capped_df = df.copy()
        numerical_cols = self._get_schema_numerical_columns(capped_df)

        for col in numerical_cols:
            q1 = capped_df[col].quantile(0.25)
            q3 = capped_df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            capped_df[col] = np.where(capped_df[col] < lower_bound, lower_bound, capped_df[col])
            capped_df[col] = np.where(capped_df[col] > upper_bound, upper_bound, capped_df[col])

        return capped_df

    def post_split_train_processing(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.cap_outliers(X_train), X_test

    def build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        logging.info("Building vehicle-maintenance preprocessor")
        nominal_features = [
            col for col in self.data_transformation_config.nominal_features if col in X_train.columns
        ]
        ordinal_features = {
            col: values
            for col, values in self.data_transformation_config.ordinal_features.items()
            if col in X_train.columns
        }
        ordinal_cols = list(ordinal_features.keys())
        numerical_features = self._get_schema_numerical_columns(X_train)

        transformers = []
        if ordinal_cols:
            transformers.append(
                (
                    "ord",
                    OrdinalEncoder(
                        categories=[ordinal_features[col] for col in ordinal_cols],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    ordinal_cols,
                )
            )
        if nominal_features:
            transformers.append(
                (
                    "nom",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    nominal_features,
                )
            )
        if numerical_features:
            transformers.append(("num", RobustScaler(), numerical_features))

        if not transformers:
            raise ValueError("No vehicle-maintenance transformers could be built from the current schema.")

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def should_apply_smote(self) -> bool:
        return True


class HyundaiCarsDataTransformation(BaseDataTransformation):
    """Minimal schema-driven preprocessing for the Hyundai dataset."""

    def encode_target(self, y_train: pd.Series, y_test: pd.Series) -> tuple[pd.Series, pd.Series]:
        logging.info("Encoding Hyundai maintenance-type target labels")
        label_encoder = LabelEncoder()
        y_train_encoded = pd.Series(
            label_encoder.fit_transform(y_train.astype(str)),
            index=y_train.index,
            name=y_train.name,
        )
        y_test_encoded = pd.Series(
            label_encoder.transform(y_test.astype(str)),
            index=y_test.index,
            name=y_test.name,
        )

        classes_path = os.path.join(
            self.data_transformation_config.training_pipeline_config.artifact_dir,
            "target_classes.json",
        )
        os.makedirs(os.path.dirname(classes_path), exist_ok=True)
        with open(classes_path, "w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "target_column": self.target_column,
                    "classes": label_encoder.classes_.tolist(),
                },
                file_obj,
                indent=2,
            )
        logging.info(f"Saved Hyundai target classes -> {classes_path}")
        return y_train_encoded, y_test_encoded

    def build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        logging.info("Building Hyundai cars preprocessor")
        categorical_features = self._get_schema_categorical_columns(X_train)
        numerical_features = self._get_schema_numerical_columns(X_train)

        transformers = []
        if categorical_features:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                )
            )
        if numerical_features:
            transformers.append(("num", RobustScaler(), numerical_features))

        if not transformers:
            raise ValueError("No Hyundai cars transformers could be built from the current schema.")

        return ColumnTransformer(transformers=transformers, remainder="drop")


class EngineDataTransformation(BaseDataTransformation):
    """Numeric-only preprocessing for engine condition data."""

    def build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        logging.info("Building engine-data preprocessor")
        numerical_features = self._get_schema_numerical_columns(X_train)
        if not numerical_features:
            raise ValueError("Engine dataset requires at least one numerical feature.")
        return ColumnTransformer(
            transformers=[("num", RobustScaler(), numerical_features)],
            remainder="drop",
        )

    def should_apply_smote(self) -> bool:
        return True


class DataTransformation:
    """Compatibility dispatcher that returns the right dataset-specific transformer."""

    PROFILE_CLASS_MAP = {
        "vehicle_maintenance": VehicleMaintenanceDataTransformation,
        "cars_hyundai": HyundaiCarsDataTransformation,
        "engine_data": EngineDataTransformation,
    }
    TARGET_CLASS_MAP = {
        "needmaintenance": VehicleMaintenanceDataTransformation,
        "anomalyindication": HyundaiCarsDataTransformation,
        "maintenancetype": HyundaiCarsDataTransformation,
        "enginecondition": EngineDataTransformation,
    }

    def __new__(
        cls,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        schema_config = read_yaml_file(file_path=data_transformation_config.schema_file_path) or {}
        profile_name = getattr(
            data_transformation_config.training_pipeline_config,
            "profile_name",
            "",
        )
        selected_class = cls.PROFILE_CLASS_MAP.get(profile_name)

        if selected_class is None:
            normalized_target = BaseDataTransformation._normalize_col_name(
                data_transformation_config.target_column or schema_config.get("target_column", "")
            )
            selected_class = cls.TARGET_CLASS_MAP.get(
                normalized_target,
                VehicleMaintenanceDataTransformation,
            )

        logging.info(
            f"Using {selected_class.__name__} for profile='{profile_name}' "
            f"schema='{data_transformation_config.schema_file_path}'."
        )
        return selected_class(
            data_ingestion_artifact=data_ingestion_artifact,
            data_transformation_config=data_transformation_config,
            data_validation_artifact=data_validation_artifact,
        )
