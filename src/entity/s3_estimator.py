from pandas import DataFrame

from src.cloud_storage.aws_storage import SimpleStorageService
from src.logger import logging


class Proj1Estimator:
    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model = None

    def is_model_present(self, model_path: str) -> bool:
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except Exception as e:
            logging.warning(f"is_model_present check failed: {e}")
            return False

    def load_model(self):
        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        self.s3.upload_file(
            from_file,
            to_filename=self.model_path,
            bucket_name=self.bucket_name,
            remove=remove,
        )

    def predict(self, X):
        if self.loaded_model is None:
            self.loaded_model = self.load_model()
        if hasattr(self.loaded_model, "predict"):
            return self.loaded_model.predict(X)
        return None
