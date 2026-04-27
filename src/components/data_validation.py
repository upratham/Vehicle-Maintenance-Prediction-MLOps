import json
import os

import pandas as pd
from pandas import DataFrame

from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema_config = read_yaml_file(file_path=self.data_validation_config.schema_file_path)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        status = len(dataframe.columns) == len(self._schema_config["columns"])
        logging.info(f"Is required column present: [{status}]")
        return status

    def is_column_exist(self, df: DataFrame) -> bool:
        cols = set(df.columns)
        missing_num = [c for c in self._schema_config["numerical_columns"] if c not in cols]
        missing_cat = [c for c in self._schema_config["categorical_columns"] if c not in cols]
        if missing_num:
            logging.info(f"Missing numerical columns: {missing_num}")
        if missing_cat:
            logging.info(f"Missing categorical columns: {missing_cat}")
        return not (missing_num or missing_cat)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        return pd.read_csv(file_path)

    def initiate_data_validation(self) -> DataValidationArtifact:
        validation_error_msg = ""
        logging.info("Starting data validation")
        df = DataValidation.read_data(file_path=self.data_ingestion_artifact.feature_store_file_path)

        if not self.validate_number_of_columns(dataframe=df):
            validation_error_msg += "Columns are missing in dataframe. "

        if not self.is_column_exist(df=df):
            validation_error_msg += "Columns are missing in dataframe. "

        validation_status = len(validation_error_msg) == 0
        data_validation_artifact = DataValidationArtifact(
            validation_status=validation_status,
            message=validation_error_msg,
            validation_report_file_path=self.data_validation_config.validation_report_file_path,
            profile_name=self.data_ingestion_artifact.profile_name,
        )

        report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
        os.makedirs(report_dir, exist_ok=True)
        with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
            json.dump(
                {"validation_status": validation_status, "message": validation_error_msg.strip()},
                report_file,
                indent=4,
            )

        logging.info(f"Data validation complete: status={validation_status}")
        return data_validation_artifact
