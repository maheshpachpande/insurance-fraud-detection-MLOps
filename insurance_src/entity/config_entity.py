import os
from insurance_src.constants import *
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()



# ----------------- Training Pipeline Config -----------------
@dataclass(frozen=True)
class TrainingPipelineConfig:
    """Global settings for the training pipeline."""
    name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


# ----------------- MongoDB Config -----------------
@dataclass(frozen=True)
class MongoDBConfig:
    """Holds MongoDB connection settings."""
    uri: str = os.getenv("MONGODB_URL") or ""
    default_db: str = DB_NAME

    def __post_init__(self):
        if not self.uri:
            raise ValueError("MongoDB URL is not set in environment variables.")



# ----------------- Data Ingestion Config -----------------
@dataclass(frozen=True)
class DataIngestionConfig:
    """Holds paths and parameters for data ingestion stage."""
    
    # Base directory for ingestion artifacts
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR)
    
    # Feature store paths
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, RAW_DATA_FILE)
    
    # Ingested datasets
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_DATA_FILE)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_DATA_FILE)
    
    # Ingestion parameters
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = COLLECTION_NAME


    

@dataclass(frozen=True)
class DataValidationConfig:
    """Holds paths and filenames for data validation stage."""

    # Base directory for data validation artifacts
    validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR)
    validation_output_path: str = os.path.join(validation_dir, VALIDATION_OUTPUT)

    # Directory containing validated datasets
    validated_data_dir: str = os.path.join(validation_dir, DATA_VALIDATION_VALID_DIR)
    validated_train_file_path: str = os.path.join(validated_data_dir, TRAIN_DATA_FILE)
    validated_test_file_path: str = os.path.join(validated_data_dir, TEST_DATA_FILE)

    # Directory containing drift reports
    drift_report_dir: str = os.path.join(validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR)
    prior_drift_report_file_path: str = os.path.join(drift_report_dir, DATA_VALIDATION_PRIOR_REPORT_FILE)
    drift_report_file_path: str = os.path.join(drift_report_dir, DATA_VALIDATION_REPORT_FILE)
