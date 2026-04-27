import os

import pandas as pd
from pandas import DataFrame

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.data_access.proj1_data import Proj1Data


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config

    def export_data_into_feature_store(self) -> DataFrame:
        """Export data from local CSV cache if present, else MongoDB; persist to the feature store path."""
        feature_store_file_path = self.data_ingestion_config.feature_store_file_path
        collection_name = self.data_ingestion_config.collection_name
        local_csv_path = os.path.join("data", f"{collection_name}.csv")

        if os.path.exists(local_csv_path) and os.path.getsize(local_csv_path) > 0:
            logging.info(f"[{collection_name}] Loading from local CSV {local_csv_path}; skipping MongoDB.")
            dataframe = pd.read_csv(local_csv_path)
        else:
            logging.info(f"Exporting data from mongodb (collection={collection_name})")
            my_data = Proj1Data(database_name=self.data_ingestion_config.database_name)
            dataframe = my_data.export_collection_as_dataframe(collection_name=collection_name)

        logging.info(f"Shape of dataframe: {dataframe.shape}")
        os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
        dataframe.to_csv(feature_store_file_path, index=False, header=True)
        return dataframe

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        dataframe = self.export_data_into_feature_store()
        return DataIngestionArtifact(
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
            profile_name=getattr(self.data_ingestion_config.training_pipeline_config, "profile_name", ""),
        )
