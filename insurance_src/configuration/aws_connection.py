import os
import boto3
import threading
from insurance_src.constants import (
    AWS_SECRET_ACCESS_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_REGION_NAME,
)
from insurance_src.exceptions import CustomException


class S3Client:
    """
    Singleton wrapper around boto3 client and resource.
    
    ✅ Ensures AWS S3 connections are created once per process
    ✅ Thread-safe for multi-worker environments
    ✅ Uses environment variables for credentials
    """

    # Class-level singletons
    _s3_client = None
    _s3_resource = None
    _lock = threading.Lock()  # thread safety lock

    def __init__(self, region_name: str = AWS_REGION_NAME):
        """
        Initialize AWS S3 client and resource using environment variables.
        Ensures only one instance of boto3 client/resource exists.
        """

        # Double-checked locking pattern
        if not S3Client._s3_client or not S3Client._s3_resource:
            with S3Client._lock:  # prevent race conditions
                if not S3Client._s3_client or not S3Client._s3_resource:
                    
                    # 🔑 Fetch credentials from environment variables
                    access_key_id = os.getenv(AWS_ACCESS_KEY_ID)
                    secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY)

                    # 🚨 Raise meaningful errors if env vars missing
                    if not access_key_id:
                        raise CustomException(
                            f"Environment variable {AWS_ACCESS_KEY_ID} is not set."
                        )
                    if not secret_access_key:
                        raise CustomException(
                            f"Environment variable {AWS_SECRET_ACCESS_KEY} is not set."
                        )

                    # 🌐 Create boto3 resource (high-level)
                    S3Client._s3_resource = boto3.resource(
                        "s3",
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=secret_access_key,
                        region_name=region_name,
                    )

                    # ⚡ Create boto3 client (low-level API)
                    S3Client._s3_client = boto3.client(
                        "s3",
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=secret_access_key,
                        region_name=region_name,
                    )

        # Instance-level references to class-level singletons
        self.s3_resource = S3Client._s3_resource
        self.s3_client = S3Client._s3_client
