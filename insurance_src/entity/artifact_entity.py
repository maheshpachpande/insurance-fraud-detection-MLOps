from dataclasses import dataclass



# ----------------- Data Ingestion Artifact -----------------
@dataclass(frozen=True)
class DataIngestionArtifact:
    """Holds file paths for the data ingestion stage."""
    
    raw_file_path: str
    trained_file_path: str
    test_file_path: str

    def __str__(self):
        return (
            f"\n📂 Data Ingestion Artifact\n"
            f"---------------------------------\n"
            f"📝 Raw Data File    : {self.raw_file_path}\n"
            f"🎯 Training File    : {self.trained_file_path}\n"
            f"🧪 Testing File     : {self.test_file_path}\n"
        )



# ----------------- Data Validation Artifact -----------------
@dataclass(frozen=True)
class DataValidationArtifact:
    """Holds validation status and validated file paths."""
    
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    drift_report_file_path: str
    
    def __str__(self):
        return (
            f"\n📂 Data Validation Artifact\n"
            f"---------------------------------\n"
            f"📝 Validated Status     : {self.validation_status}\n"
            f"📝 Validated Train File: {self.valid_train_file_path}\n"  
            f"📝 Validated Test File : {self.valid_test_file_path}\n"
            f"📝 Drift Report File   : {self.drift_report_file_path}\n"
        )
        

@dataclass(frozen=True)
class DataTransformationArtifact:
    """
    Holds file paths for transformed datasets and preprocessing object.
    """

    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

    def __str__(self):
        return (
            f"\n📂 Data Transformation Artifact\n"
            f"---------------------------------\n"
            f"📝 Preprocessor Object File : {self.transformed_object_file_path}\n"
            f"📝 Transformed Train File   : {self.transformed_train_file_path}\n"
            f"📝 Transformed Test File    : {self.transformed_test_file_path}\n"
        )
        
        
from dataclasses import dataclass
from typing import Optional


# ---------------------------
# Classification Metrics Artifact
# ---------------------------
@dataclass(frozen=True)
class ClassificationMetricArtifact:
    """
    Artifact to store classification evaluation metrics.
    Immutable once created.
    """
    f1_score: float        # Harmonic mean of precision and recall
    precision_score: float # Correct positive predictions / total predicted positives
    recall_score: float    # Correct positive predictions / total actual positives

    def __str__(self) -> str:
        return (
            f"\nClassification Metrics:\n"
            f"  F1 Score      : {self.f1_score:.4f}\n"
            f"  Precision     : {self.precision_score:.4f}\n"
            f"  Recall        : {self.recall_score:.4f}\n"
        )


# ---------------------------
# Model Trainer Artifact
# ---------------------------
@dataclass(frozen=True)
class ModelTrainerArtifact:
    """
    Artifact produced after model training.
    Stores trained model path and evaluation metrics.
    Immutable once created.
    """
    trained_model_file_path: str                                # Path where the trained model is saved
    test_metric_artifact: ClassificationMetricArtifact          # Metrics on test dataset
    train_metric_artifact: Optional[ClassificationMetricArtifact] = None  # (Optional) metrics on training dataset

    def __str__(self) -> str:
        result = (
            f"\nModel Trainer Artifact:\n"
            f"  Trained Model Path : {self.trained_model_file_path}\n"
        )
        if self.train_metric_artifact:
            result += f"  Train Metrics      : {self.train_metric_artifact}"
        result += f"  Test Metrics       : {self.test_metric_artifact}"
        return result


# ---------------------------
# Model Evolution Artifact
# ---------------------------

@dataclass
class ModelEvaluationArtifact:
    # Whether the newly trained model is accepted based on evaluation
    is_model_accepted: bool
    # Change in accuracy (or metric) compared to previous model
    changed_accuracy: float
    # Path to the model stored in S3
    s3_model_path: str
    # Local path to the trained model
    trained_model_path: str

    def __str__(self) -> str:
        return (
            f"ModelEvaluationArtifact(\n"
            f"  is_model_accepted = {self.is_model_accepted},\n"
            f"  changed_accuracy  = {self.changed_accuracy:.4f},\n"
            f"  s3_model_path     = '{self.s3_model_path}',\n"
            f"  trained_model_path= '{self.trained_model_path}'\n"
            f")"
        )


# ---------------------------
# Model Pusher Artifact
# ---------------------------


@dataclass
class ModelPusherArtifact:
    # Name of the S3 bucket where the model is pushed
    bucket_name: str
    # Full S3 path of the pushed model
    s3_model_path: str

    def __str__(self) -> str:
        return (
            f"ModelPusherArtifact(\n"
            f"  bucket_name   = '{self.bucket_name}',\n"
            f"  s3_model_path = '{self.s3_model_path}'\n"
            f")"
        )


# # ---------------------------
# # Model Pusher Artifact
# # ---------------------------

# @dataclass
# class ModelPusherConfig:
#     bucket_name: str = MODEL_BUCKET_NAME
#     s3_model_key_path: str = MODEL_FILE_NAME

#     def __str__(self) -> str:
#         return (
#             f"ModelPusherConfig(\n"
#             f"  bucket_name      = '{self.bucket_name}',\n"
#             f"  s3_model_key_path = '{self.s3_model_key_path}'\n"
#             f")"
#         )


# # ---------------------------
# # Predictor Artifact
# # ---------------------------
# @dataclass
# class PredictorConfig:
#     model_file_path: str = MODEL_FILE_NAME
#     model_bucket_name: str = MODEL_BUCKET_NAME

#     def __str__(self) -> str:
#         return (
#             f"PredictorConfig(\n"
#             f"  model_file_path    = '{self.model_file_path}',\n"
#             f"  model_bucket_name  = '{self.model_bucket_name}'\n"
#             f")"
#         )
