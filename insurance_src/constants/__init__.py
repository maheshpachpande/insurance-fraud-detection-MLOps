"""
constants.py
-------------
Global constants for the Insurance Fraud Detection pipeline.
Follow UPPER_SNAKE_CASE naming, type hints, and env-based secure configuration.
"""

import os
from datetime import date

# =====================================================
# 📌 Database
# =====================================================
DB_NAME: str = "insurance"
COLLECTION_NAME: str = "data"

# =====================================================
# 📌 General / Global
# =====================================================
TARGET_COLUMN: str = "fraud_reported"
PIPELINE_NAME: str = "insurance"
ARTIFACT_DIR: str = "artifact"

RAW_DATA_FILE: str = "raw.csv"
TRAIN_DATA_FILE: str = "train.csv"
TEST_DATA_FILE: str = "test.csv"

SAVED_MODEL_DIR: str = "saved_models"
MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSOR_FILE_NAME: str = "preprocessing.pkl"

CURRENT_YEAR: int = date.today().year

VALIDATION_OUTPUT: str = "validation.yaml"

SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLS_KEY: str = "drop_columns"

FEATURE_ENGINEERING_CONFIG_FILE: str = "config/feature_engineering.yaml"

# =====================================================
# 📌 AWS (Env-Based)
# =====================================================
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION_NAME: str = os.getenv("AWS_REGION_NAME", "us-east-1")
MODEL_BUCKET_NAME: str = os.getenv("MODEL_BUCKET_NAME", "insurance23")

# =====================================================
# 📌 Data Ingestion
# =====================================================
DATA_INGESTION_COLLECTION: str = "data"
DATA_INGESTION_DIR: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

# =====================================================
# 📌 Data Validation
# =====================================================
DATA_VALIDATION_DIR: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

DATA_VALIDATION_PRIOR_REPORT_FILE: str = "probability_drift_target_column.yaml"
DATA_VALIDATION_REPORT_FILE: str = "dataset_drift.yaml"

DATA_VALIDATION_VALIDATED_PATH: str = os.path.join(
    ARTIFACT_DIR, DATA_VALIDATION_DIR, DATA_VALIDATION_VALID_DIR
)

# =====================================================
# 📌 Data Transformation
# =====================================================
DATA_TRANSFORMATION_DIR: str = "data_transformation"
DATA_TRANSFORMATION_OUTPUT_DIR: str = "transformed_numpy_array"
DATA_TRANSFORMATION_OBJECT_DIR: str = "transformed_binary_object"

# =====================================================
# 📌 Model Training
# =====================================================
MODEL_TRAINER_DIR: str = "model_trainer"
MODEL_TRAINER_OUTPUT_DIR: str = "trained_model"
MODEL_TRAINER_ARTIFACT_FILE: str = "model_trainer_artifact.yaml"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.5
MODEL_TRAINER_OVERFIT_THRESHOLD: float = 0.8
MODEL_TRAINER_CONFIG_FILE: str = os.path.join("config", "model.yaml")

# =====================================================
# 📌 Model Evaluation
# =====================================================
MODEL_EVALUATION_DIR: str = "model_evaluation"
MODEL_EVALUATION_SCORE_CHANGE_THRESHOLD: float = 0.5
MODEL_EVALUATION_REPORT_FILE: str = "report.yaml"
MODEL_EVALUATION_ARTIFACT_FILE: str = "model_evaluation_artifact.yaml"

# =====================================================
# 📌 Model Pusher
# =====================================================
MODEL_PUSHER_DIR: str = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR: str = SAVED_MODEL_DIR

# =====================================================
# 📌 Application Server
# =====================================================
APP_HOST: str = "0.0.0.0"
APP_PORT: int = int(os.getenv("APP_PORT", 8080))
