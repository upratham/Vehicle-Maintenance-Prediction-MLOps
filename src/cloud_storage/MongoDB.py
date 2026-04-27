from pathlib import Path
import pandas as pd
from src.constants import DATA_DIR
from src.logger import logging
import pymongo
import os

def get_collection_name_from_csv(csv_path: str) -> str:
    """Return MongoDB collection name from CSV filename (without .csv)."""
    csv_file = Path(csv_path)
    return csv_file.stem

def push_csv_to_mongodb(
    csv_path: str,
    db_name: str = "605_Project_Data",
    connection_url: str | None = None,
    push: bool = True,
):
    """
    Push a CSV file to MongoDB when push=True.
    Collection name is derived from CSV filename without extension.
    When push=False, only collection name is returned (no DB write).
    """
    csv_file = Path(csv_path)
    logging.info(f"Starting CSV -> MongoDB flow for file: {csv_file}")
    if not csv_file.exists():
        logging.error(f"CSV file not found: {csv_file}")
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    collection_name = get_collection_name_from_csv(csv_path)
    logging.info(f"Resolved collection name: {collection_name}")

    if not push:
        logging.info("Push disabled (preview mode). Returning collection name without insert.")
        return {"collection": collection_name, "inserted_count": 0, "pushed": False}

    connection_url = connection_url or os.getenv("CONNECTION_URL")
    if not connection_url:
        logging.error("Mongo connection URL missing in function argument and environment.")
        raise ValueError("Mongo connection URL missing. Set CONNECTION_URL or pass connection_url.")

    df = pd.read_csv(csv_file)
    records = df.to_dict(orient="records")
    logging.info(f"Loaded {len(records)} records from {csv_file.name}")

    if not records:
        logging.warning(f"No rows found in {csv_file.name}. Nothing inserted.")
        return {"collection": collection_name, "inserted_count": 0, "pushed": False}

    logging.info(f"Connecting to MongoDB and inserting into {db_name}.{collection_name}")
    client = pymongo.MongoClient(connection_url)
    collection = client[db_name][collection_name]
    result = collection.insert_many(records)
    logging.info(
        f"Insert successful for {db_name}.{collection_name}. Inserted {len(result.inserted_ids)} records."
    )

    return {
        "database": db_name,
        "collection": collection_name,
        "inserted_count": len(result.inserted_ids),
        "pushed": True,
    }

def main():
    csv_paths = [f"{DATA_DIR}/{file}" for file in os.listdir(DATA_DIR) if file.endswith(".csv")]
    logging.info(f"Found {len(csv_paths)} CSV files in {DATA_DIR}")
    for csv_path in csv_paths:
        try:
            result = push_csv_to_mongodb(csv_path)
            logging.info(f"Result for {csv_path}: {result}")
            print(result)
        except Exception as exc:
            logging.exception(f"Failed to upload CSV {csv_path}: {exc}")

if __name__ == "__main__":
    main()