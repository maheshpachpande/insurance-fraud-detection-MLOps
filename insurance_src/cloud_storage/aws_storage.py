import boto3
from insurance_src.configuration.aws_connection import S3Connection
from io import StringIO, BytesIO
from typing import Union, List, cast
import os
import sys
from pandas import DataFrame, read_csv
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import S3ServiceResource, Bucket, Object
from mypy_boto3_s3.client import S3Client
from insurance_src.logger import logging
from insurance_src.exceptions import CustomException
import pickle


class SimpleStorageService:
    s3_resource: S3ServiceResource
    s3_client: S3Client

    def __init__(self):
        try:
            s3_conn = S3Connection()
            self.s3_resource = cast(S3ServiceResource, s3_conn._s3_resource)
            self.s3_client = cast(S3Client, s3_conn._s3_client)
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    # ---------- S3 Utilities ----------

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = list(bucket.objects.filter(Prefix=s3_key))
            return len(file_objects) > 0
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    def get_bucket(self, bucket_name: str) -> Bucket:
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    def get_file_object(self, filename: str, bucket_name: str) -> Union[Object, List[Object]]:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = list(bucket.objects.filter(Prefix=filename))
            if not file_objects:
                raise FileNotFoundError(f"No object found with prefix '{filename}' in bucket '{bucket_name}'")
            return file_objects[0] if len(file_objects) == 1 else file_objects
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    # ---------- Read Objects ----------

    def read_object_bytes(self, bucket_name: str, object_name: str) -> BytesIO:
        """Read S3 object as bytes (for pickle, upload_fileobj, etc.)"""
        try:
            obj: Object = self.s3_resource.Object(bucket_name, object_name)
            body = obj.get()["Body"].read()
            return BytesIO(body)
        except ClientError as e:
            raise CustomException(e, sys.exc_info())

    def read_object_text(self, bucket_name: str, object_name: str) -> StringIO:
        """Read S3 object as text (for CSV/text)"""
        try:
            obj: Object = self.s3_resource.Object(bucket_name, object_name)
            body = obj.get()["Body"].read().decode("utf-8")
            return StringIO(body)
        except ClientError as e:
            raise CustomException(e, sys.exc_info())

    # ---------- Model Operations ----------

    def load_model(self, model_path: str, bucket_name: str):
        try:
            model_bytes = self.read_object_bytes(bucket_name, model_path)
            model = pickle.load(model_bytes)
            return model
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    # ---------- Folder/File Operations ----------

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        try:
            self.s3_resource.Object(bucket_name, folder_name + "/").load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_name + "/")
            else:
                raise CustomException(e, sys.exc_info())

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        try:
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            if remove:
                os.remove(from_filename)
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    def upload_df_as_csv(self, df: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str):
        try:
            df.to_csv(local_filename, index=False)
            self.upload_file(local_filename, bucket_filename, bucket_name)
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    # ---------- DataFrame Operations ----------

    def get_df_from_object(self, object_: Union[Object, BytesIO]) -> DataFrame:
        try:
            if isinstance(object_, Object):
                content = self.read_object_text(object_.bucket_name, object_.key)
            elif isinstance(object_, BytesIO):
                content = StringIO(object_.read().decode("utf-8"))
            else:
                raise TypeError("Unsupported object type for conversion to DataFrame")
            df = read_csv(content, na_values="na")
            return df
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        try:
            obj = self.get_file_object(filename, bucket_name)
            return self.get_df_from_object(obj)
        except Exception as e:
            raise CustomException(e, sys.exc_info())
