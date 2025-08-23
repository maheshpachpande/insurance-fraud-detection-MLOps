import os
import sys
import pickle
from io import StringIO
from typing import Union, List

import boto3
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv
from mypy_boto3_s3.service_resource import Bucket

from insurance_src.configuration.aws_connection import S3Client
from insurance_src.logger import logging
from insurance_src.exceptions import CustomException

from io import StringIO

# ---------------------------
# Abstraction Layer (DIP)
# ---------------------------
class StorageService:
    """Interface for any storage service (S3, GCS, local)."""

    def upload_file(self, local_path: str, bucket_name: str, remote_path: str, remove_local: bool = True):
        """Upload a local file to remote storage."""
        raise NotImplementedError

    def download_file(self, bucket_name: str, remote_path: str, local_path: str):
        """Download a file from remote storage to local path."""
        raise NotImplementedError

    def object_exists(self, bucket_name: str, key: str) -> bool:
        """Check if object exists in remote storage."""
        raise NotImplementedError


# ---------------------------
# Concrete Implementation for S3 (SRP, DIP)
# ---------------------------
class S3StorageService(StorageService):
    """Amazon S3 implementation of StorageService."""

    def __init__(self):
        """Initialize S3 client and resource using S3Client singleton."""
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    # ---------- Bucket Operations ----------
    def get_bucket(self, bucket_name: str) -> Bucket:
        """Return a boto3 Bucket object."""
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            raise CustomException(e)

    def object_exists(self, bucket_name: str, key: str) -> bool:
        """Check if a key exists in the given bucket."""
        try:
            bucket = self.get_bucket(bucket_name)
            return any(bucket.objects.filter(Prefix=key))
        except Exception as e:
            raise CustomException(e)

    def create_folder(self, bucket_name: str, folder_name: str) -> None:
        """Create a folder in S3 if it does not exist."""
        try:
            self.s3_resource.Object(bucket_name, folder_name).load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.s3_client.put_object(Bucket=bucket_name, Key=f"{folder_name}/")
            else:
                raise

    # ---------- File Operations ----------
    def upload_file(self, local_path: str, bucket_name: str, remote_path: str, remove_local: bool = True):
        """Upload a file to S3 and optionally remove local copy."""
        try:
            self.s3_resource.meta.client.upload_file(local_path, bucket_name, remote_path)
            logging.info(f"Uploaded {local_path} to s3://{bucket_name}/{remote_path}")

            if remove_local:
                os.remove(local_path)
                logging.info(f"Removed local file {local_path}")
        except Exception as e:
            raise CustomException(e)

    def download_file(self, bucket_name: str, remote_path: str, local_path: str):
        """Download a file from S3 to local path."""
        try:
            self.s3_resource.meta.client.download_file(bucket_name, remote_path, local_path)
            logging.info(f"Downloaded s3://{bucket_name}/{remote_path} to {local_path}")
        except Exception as e:
            raise CustomException(e)

    # ---------- Object Operations ----------
    @staticmethod
    def read_object(s3_object, decode: bool = True, as_stream: bool = False) -> Union[StringIO, str, bytes]:
        """Read content from S3 object. Can return as string, bytes, or StringIO stream."""
        try:
            body = s3_object.get()["Body"].read()
            if decode:
                body = body.decode()
            return StringIO(body) if as_stream else body
        except Exception as e:
            raise CustomException(e)

    def get_file_object(self, bucket_name: str, key: str) -> Union[List[object], object]:
        """Return S3 object(s) matching the given key."""
        try:
            bucket = self.get_bucket(bucket_name)
            objs = [obj for obj in bucket.objects.filter(Prefix=key)]
            return objs[0] if len(objs) == 1 else objs
        except Exception as e:
            raise CustomException(e)


# ---------------------------
# DataFrame Handler (SRP)
# ---------------------------
class S3DataFrameHandler:
    """Handles reading and writing pandas DataFrames to/from S3."""

    def __init__(self, storage: S3StorageService):
        self.storage = storage

    def upload_dataframe(self, df: DataFrame, local_path: str, bucket_name: str, remote_path: str):
        """Upload DataFrame as CSV to S3."""
        df.to_csv(local_path, index=False)
        self.storage.upload_file(local_path, bucket_name, remote_path)

    def read_csv(self, bucket_name: str, key: str) -> DataFrame:
        obj = self.storage.get_file_object(bucket_name, key)
        content = self.storage.read_object(obj, decode=True)  # returns str
        return read_csv(StringIO(content), na_values="na")  # force file-like



# ---------------------------
# Model Handler (SRP)
# ---------------------------
class S3ModelHandler:
    """Handles saving and loading machine learning models to/from S3."""

    def __init__(self, storage: S3StorageService):
        self.storage = storage

    def load_model(self, bucket_name: str, key: str) -> object:
        """Load a pickled model from S3."""
        obj = self.storage.get_file_object(bucket_name, key)
        model_bytes = self.storage.read_object(obj, decode=False)
        return pickle.loads(model_bytes)

    def save_model(self, model: object, local_path: str, bucket_name: str, remote_path: str):
        """Save a model locally and upload it to S3."""
        with open(local_path, "wb") as f:
            pickle.dump(model, f)
        self.storage.upload_file(local_path, bucket_name, remote_path)
