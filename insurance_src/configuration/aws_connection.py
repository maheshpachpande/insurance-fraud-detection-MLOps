import os
import boto3

from boto3.resources.base import ServiceResource
from typing import Optional, List
from abc import ABC, abstractmethod
from insurance_src.constants import AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_ACCESS_KEY_ID_ENV_KEY, REGION_NAME

# ---------------------------
# 1️⃣ AWS Credentials Manager
# ---------------------------
class AWSCredentials:
    """Fetch AWS credentials from environment variables"""
    def __init__(self):
        self.access_key = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
        self.secret_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
        if not self.access_key or not self.secret_key:
            raise EnvironmentError(
                f"AWS credentials not found. Ensure {AWS_ACCESS_KEY_ID_ENV_KEY} "
                f"and {AWS_SECRET_ACCESS_KEY_ENV_KEY} are set."
            )

# ---------------------------
# 2️⃣ S3 Connection Manager
# ---------------------------


class S3Connection:
    _s3_client = None
    _s3_resource: Optional[ServiceResource] = None

    def __init__(self, credentials: Optional[AWSCredentials] = None, region_name=REGION_NAME):
        if credentials is None:
            credentials = AWSCredentials()

        if S3Connection._s3_client is None or S3Connection._s3_resource is None:
            S3Connection._s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                region_name=region_name
            )
            S3Connection._s3_resource = boto3.resource(
                's3',
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                region_name=region_name
            )

        self.client = S3Connection._s3_client
        self.resource: ServiceResource = S3Connection._s3_resource


# ---------------------------
# 3️⃣ S3 Operations Interface
# ---------------------------
class S3ServiceInterface(ABC):
    """Abstract interface for S3 operations"""
    @abstractmethod
    def upload_file(self, file_path: str, bucket_name: str, object_name: str):
        pass

    @abstractmethod
    def download_file(self, bucket_name: str, object_name: str, dest_path: str):
        pass

    @abstractmethod
    def list_files(self, bucket_name: str, prefix: str = "") -> List[str]:
        """Return a list of file keys"""
        pass

    @abstractmethod
    def delete_file(self, bucket_name: str, object_name: str):
        pass

# ---------------------------
# 4️⃣ S3 Service Implementation
# ---------------------------
class S3Service(S3ServiceInterface):
    """Concrete implementation of S3 operations"""
    def __init__(self, connection: S3Connection):
        self.s3_client = connection.client
        self.s3_resource = connection.resource

    def upload_file(self, file_path: str, bucket_name: str, object_name: str):
        self.s3_client.upload_file(file_path, bucket_name, object_name)

    def download_file(self, bucket_name: str, object_name: str, dest_path: str):
        self.s3_client.download_file(bucket_name, object_name, dest_path)

    def list_files(self, bucket_name: str, prefix: str = ""):
        bucket = self.s3_resource.Bucket(bucket_name) # type: ignore
        return [obj.key for obj in bucket.objects.filter(Prefix=prefix)]

    def delete_file(self, bucket_name: str, object_name: str):
        self.s3_client.delete_object(Bucket=bucket_name, Key=object_name)
