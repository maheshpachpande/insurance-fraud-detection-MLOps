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