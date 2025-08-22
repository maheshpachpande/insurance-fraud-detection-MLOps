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