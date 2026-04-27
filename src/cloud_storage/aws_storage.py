import os
import pickle
from io import StringIO
from typing import Union, List

from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket
from pandas import DataFrame, read_csv

from src.configuration.aws_connection import S3Client
from src.logger import logging

_S3_DELETE_BATCH = 1000


class SimpleStorageService:
    """Convenience wrapper around boto3 for the operations this project actually uses."""

    def __init__(self):
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        bucket = self.get_bucket(bucket_name)
        return any(True for _ in bucket.objects.filter(Prefix=s3_key))

    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        body = object_name.get()["Body"].read()
        if decode:
            body = body.decode()
        return StringIO(body) if make_readable else body

    def get_bucket(self, bucket_name: str) -> Bucket:
        return self.s3_resource.Bucket(bucket_name)

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        bucket = self.get_bucket(bucket_name)
        file_objects = list(bucket.objects.filter(Prefix=filename))
        return file_objects[0] if len(file_objects) == 1 else file_objects

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        model_file = f"{model_dir}/{model_name}" if model_dir else model_name
        file_object = self.get_file_object(model_file, bucket_name)
        model_obj = self.read_object(file_object, decode=False)
        logging.info("Production model loaded from S3 bucket.")
        return pickle.loads(model_obj)

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        try:
            self.s3_resource.Object(bucket_name, folder_name).load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.s3_client.put_object(Bucket=bucket_name, Key=f"{folder_name}/")

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        logging.info(f"Uploading {from_filename} to s3://{bucket_name}/{to_filename}")
        self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
        if remove:
            os.remove(from_filename)

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str) -> None:
        data_frame.to_csv(local_filename, index=None, header=True)
        self.upload_file(local_filename, bucket_filename, bucket_name)

    def get_df_from_object(self, object_: object) -> DataFrame:
        content = self.read_object(object_, make_readable=True)
        return read_csv(content, na_values="na")

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        csv_obj = self.get_file_object(filename, bucket_name)
        return self.get_df_from_object(csv_obj)

    def upload_directory(self, local_dir: str, s3_prefix: str, bucket_name: str) -> int:
        """Recursively upload all files in local_dir to s3_prefix. Returns number of files uploaded."""
        count = 0
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative = os.path.relpath(local_path, local_dir).replace("\\", "/")
                s3_key = f"{s3_prefix}/{relative}"
                self.s3_resource.meta.client.upload_file(local_path, bucket_name, s3_key)
                count += 1
        logging.info(f"[S3] Uploaded {count} files from {local_dir} -> s3://{bucket_name}/{s3_prefix}")
        return count

    def list_run_prefixes(self, bucket_name: str, base_prefix: str) -> list:
        """Return sorted list of immediate child prefixes under base_prefix (one level deep)."""
        prefixes = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        normalized = base_prefix.rstrip("/") + "/"
        for page in paginator.paginate(Bucket=bucket_name, Prefix=normalized, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                prefixes.append(cp["Prefix"])
        return sorted(prefixes)

    def delete_prefix(self, bucket_name: str, prefix: str) -> int:
        """Delete all S3 objects whose key starts with prefix. Returns count deleted."""
        bucket = self.get_bucket(bucket_name)
        keys = [{"Key": obj.key} for obj in bucket.objects.filter(Prefix=prefix)]
        total = 0
        for i in range(0, len(keys), _S3_DELETE_BATCH):
            batch = keys[i : i + _S3_DELETE_BATCH]
            bucket.delete_objects(Delete={"Objects": batch})
            total += len(batch)
        logging.info(f"[S3] Deleted {total} objects under s3://{bucket_name}/{prefix}")
        return total
