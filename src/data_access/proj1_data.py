from typing import Optional

import pandas as pd

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.logger import logging


class Proj1Data:
    """Export MongoDB collections as pandas DataFrames."""

    def __init__(self, database_name: Optional[str] = None) -> None:
        resolved_db_name = database_name or DATABASE_NAME or "605_Project_Data"
        self.mongo_client = MongoDBClient(database_name=resolved_db_name)

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        if database_name is None:
            collection = self.mongo_client.database[collection_name]
        else:
            collection = self.mongo_client.client[database_name][collection_name]

        logging.info(f"Fetching collection '{collection_name}' from MongoDB")
        df = pd.DataFrame(list(collection.find()))
        logging.info(f"Fetched {len(df)} records from {collection_name}")
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        return df
