import os

import certifi
import pymongo

from src.constants import DATABASE_NAME
from src.logger import logging

ca = certifi.where()


class MongoDBClient:
    """Lazy, process-cached MongoDB client. Reads CONNECTION_URL at first connect time."""

    client = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        if MongoDBClient.client is None:
            mongo_db_url = os.getenv("CONNECTION_URL")
            if not mongo_db_url:
                raise EnvironmentError("Environment variable 'CONNECTION_URL' is not set.")
            if not mongo_db_url.startswith(("mongodb://", "mongodb+srv://")):
                raise ValueError(
                    f"CONNECTION_URL has invalid URI scheme (got: {mongo_db_url[:30]!r}). "
                    "Must start with 'mongodb://' or 'mongodb+srv://'."
                )
            MongoDBClient.client = pymongo.MongoClient(
                mongo_db_url, tlsCAFile=ca, serverSelectionTimeoutMS=5000
            )

        self.client = MongoDBClient.client
        self.database = self.client[database_name]
        self.database_name = database_name
        logging.info("MongoDB connection successful.")
