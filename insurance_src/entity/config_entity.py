import os
from insurance_src.constants import *
from dataclasses import dataclass
from insurance_src.utils.main_utils import read_yaml_file

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Literal
from sklearn.base import BaseEstimator


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




@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for the data transformation stage.

    Holds directory paths and file paths for:
        - Transformed training data
        - Transformed testing data
        - Preprocessor object
    """

    # Base directory for all data transformation artifacts
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR
    )

    # Full path for the transformed training dataset (.npy)
    transformed_train_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR,
        DATA_TRANSFORMATION_OUTPUT_DIR,
        TRAIN_DATA_FILE.replace("csv", "npy"),
    )

    # Full path for the transformed testing dataset (.npy)
    transformed_test_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR,
        DATA_TRANSFORMATION_OUTPUT_DIR,
        TEST_DATA_FILE.replace("csv", "npy"),
    )

    # Full path for the saved preprocessor object
    transformed_object_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR,
        DATA_TRANSFORMATION_OBJECT_DIR,
        PREPROCESSOR_FILE_NAME,
    )
    
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration entity for the Model Training stage."""

    # Root directory for storing all model trainer artifacts
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR)

    # Full path to the final trained model file (model.pkl or equivalent)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_OUTPUT_DIR, MODEL_FILE_NAME)

    # Minimum accuracy required to accept the trained model
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE


    # Threshold to detect overfitting/underfitting
    overfitting_underfitting_threshold: float = MODEL_TRAINER_OVERFIT_THRESHOLD

    # Path to store the artifact metadata YAML file
    artifact_yaml_path: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR, MODEL_TRAINER_ARTIFACT_FILE)

    # Path to the model configuration YAML (hyperparameters, search space, etc.)
    model_config_file_path: str = MODEL_TRAINER_CONFIG_FILE
    
    


@dataclass
class ModelEvaluationConfig:
    # Minimum score change to trigger model update
    changed_threshold_score: float = MODEL_EVALUATION_SCORE_CHANGE_THRESHOLD

    # S3 bucket where the model is stored
    bucket_name: str = MODEL_BUCKET_NAME

    # S3 key path of the model file
    s3_model_key_path: str = MODEL_FILE_NAME

    # Local artifact YAML/JSON file path
    artifact_file_path: str = os.path.join(
        ARTIFACT_DIR, MODEL_EVALUATION_DIR, MODEL_EVALUATION_ARTIFACT_FILE
    )
    
    
    
@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME

    def __str__(self) -> str:
        return (
            f"ModelPusherConfig(\n"
            f"  bucket_name      = '{self.bucket_name}',\n"
            f"  s3_model_key_path = '{self.s3_model_key_path}'\n"
            f")"
        )

@dataclass
class PredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME

    def __str__(self) -> str:
        return (
            f"PredictorConfig(\n"
            f"  model_file_path    = '{self.model_file_path}',\n"
            f"  model_bucket_name  = '{self.model_bucket_name}'\n"
            f")"
        )
