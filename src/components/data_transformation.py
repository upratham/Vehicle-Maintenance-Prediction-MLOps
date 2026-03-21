import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.constants import *
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)


    def handle_duplicates_and_missing_values(self, df):
        """Handle duplicates and missing values in the DataFrame."""
        logging.info("Handling duplicates and missing values")
        df = df.drop_duplicates()
        df = df.fillna(0)
        df = df.reset_index(drop=True) # Replace missing values with 0 (or use a more sophisticated strategy)
        return df
    
    def convert_date_to_days(self, df):
        """Convert date columns to days."""
        logging.info("Converting date columns to days")
        for col in self._schema_config['date_columns']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col + "_days"] = (REFERENCE_DATE - df[col]).dt.days
                df.drop(columns=[col], inplace=True)
        return df
    
    # Cap outliers in all numerical columns (except target) using IQR method to clip extreme values within acceptable bounds
    def cap_outliers(self, df):
        """Cap outliers in numerical columns using the IQR method."""
        logging.info("Capping outliers in numerical columns")

        target_col = self._schema_config['target_column']
        numerical_cols = [
            col for col in self._schema_config['numerical_columns']
            if col != target_col
        ]

        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df
    
    def encode_categorical_variables(self, df):
        """ Encode categorical columns, compute Spearman correlation heatmap, calculate mutual information scores,
             and drop features with zero predictive power toward the target variable
        """
        logging.info("Encoding categorical variables and performing feature selection")
        df_encoded = df.copy()
        cat_cols_tmp = [c for c in self._schema_config['categorical_columns'] if c in df.columns]
        le = LabelEncoder()
        for c in cat_cols_tmp:
            df_encoded[c] = le.fit_transform(df_encoded[c].astype(str))

        spearman_corr = df_encoded.corr(method="spearman")

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
        #plt.show()

        X_tmp = df_encoded.drop(columns=["Need_Maintenance"])
        y_tmp = df_encoded["Need_Maintenance"]

        mi_scores = mutual_info_classif(X_tmp, y_tmp, discrete_features="auto", random_state=42)
        mi_series = pd.Series(mi_scores, index=X_tmp.columns).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        mi_series.plot(kind="bar", color="steelblue")
        plt.title("Mutual Information Scores vs. Need_Maintenance")
        plt.ylabel("MI Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "mi_scores.png"), dpi=300)
        #plt.show()

        zero_mi_features = mi_series[mi_series == 0].index.tolist()
        if zero_mi_features:
            df.drop(columns=zero_mi_features, inplace=True)

        logging.info(f"Surviving columns after MI filter ({len(df.columns)}): {df.columns.tolist()}") 
        
        return df

    
    def split_features_and_target(self, df):
        """Separate features (X) and target variable (y)."""
        logging.info("Splitting features and target variable")
        X = df.drop(columns=['Need_Maintenance'])
        y = df['Need_Maintenance']
        return X, y

    def data_transformation_and_scaling(self, X):
        """Apply custom transformations in specified sequence."""
        logging.info("Applying custom transformations in specified sequence")

        nominal_features = NOMINAL_FEATURES
        nominal_features = [col for col in nominal_features if col in X.columns]

        numerical_features = [
            c for c in self._schema_config['numerical_columns']
            if c != self._schema_config['target_column'] and c in X.columns
        ]

        ordinal_features =ORDINAL_FEATURES

        ordinal_features = {k: v for k, v in ordinal_features.items() if k in X.columns}
        ordinal_cols = list(ordinal_features.keys())

        ordinal_transformer = OrdinalEncoder(
            categories=[ordinal_features[col] for col in ordinal_cols],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        nominal_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        numerical_transformer = RobustScaler()

        transformers = []

        if ordinal_cols:
            transformers.append(("ord", ordinal_transformer, ordinal_cols))

        if nominal_features:
            transformers.append(("nom", nominal_transformer, nominal_features))

        if numerical_features:
            transformers.append(("num", numerical_transformer, numerical_features))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
        
        X_scaled = preprocessor.fit_transform(X)
        return X_scaled , preprocessor
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)
            logging.info("data loaded")
            # Apply transformation steps to train
            df=self.handle_duplicates_and_missing_values(df)
            df=self.convert_date_to_days(df)
            df=self.cap_outliers(df)
            df=self.encode_categorical_variables(df)
            X, y = self.split_features_and_target(df)
   
            logging.info("Custom transformations applied to data")
            logging.info("Starting data transformation")
            X_scaled, preprocessor = self.data_transformation_and_scaling(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

            logging.info("Data transformation done end to end.")

            logging.info("Applying SMOTE for handling imbalanced dataset.")
            smote = SMOTE(random_state=42)
            X_train_arr, y_train_arr = smote.fit_resample(X_train, y_train)
        
            train_arr = np.c_[X_train_arr, np.array(y_train_arr)]
            test_arr = np.c_[X_test, np.array(y_test)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving  transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e