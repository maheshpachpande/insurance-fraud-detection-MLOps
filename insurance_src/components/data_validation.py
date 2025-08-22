import os, shutil
import pandas as pd
import logging
from insurance_src.constants import SCHEMA_FILE_PATH

from insurance_src.entity.config_entity import (DataIngestionConfig,
                                                DataValidationConfig)

from insurance_src.entity.artifact_entity import (DataIngestionArtifact, 
                                                  DataValidationArtifact)

from insurance_src.utils.main_utils import (write_yaml_file, 
                                            read_yaml_file)

from insurance_src.utils.validators import SchemaValidator
from insurance_src.utils.drift_detectors import DriftDetector



class DataValidation:
    def __init__(self, ingestion_artifact:DataIngestionArtifact, 
                 validation_config: DataValidationConfig
                 ):
        
        self.ingestion_artifact = ingestion_artifact
        self.config = validation_config
        self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        self.validator = SchemaValidator(self.schema)
        self.drift_detector = DriftDetector()

    def read_data(self, path: str):
        return pd.read_csv(path)

    def initiate_data_validation(self) -> DataValidationArtifact:
        train_df = self.read_data(self.ingestion_artifact.trained_file_path)
        test_df = self.read_data(self.ingestion_artifact.test_file_path)

        schema_ok = all([
            self.validator._validate_columns(train_df),
            self.validator._validate_columns(test_df),
            self.validator._validate_numerical_columns(train_df),
            self.validator._validate_numerical_columns(test_df),
            self.validator._validate_categorical_columns(train_df),
            self.validator._validate_categorical_columns(test_df)
        ])

        drift_ok, drift_report = self.drift_detector.detect_dataset_drift(
            train_df.drop(columns=[self.schema["target_column"][0]]),
            test_df.drop(columns=[self.schema["target_column"][0]])
        )
        write_yaml_file(self.config.drift_report_file_path, drift_report)

        prior_report = self.drift_detector.detect_prior_probability_drift(
            train_df, test_df, self.schema["target_column"][0]
        )
        write_yaml_file(self.config.prior_drift_report_file_path, prior_report)

        concept_acc = self.drift_detector.detect_concept_drift(train_df.copy(), test_df.copy(), 
                                                               self.schema["target_column"][0])
        concept_ok = concept_acc >= 0.7

        os.makedirs(self.config.validated_data_dir, exist_ok=True)
        validated_train = os.path.join(self.config.validated_data_dir, "train.csv")
        validated_test = os.path.join(self.config.validated_data_dir, "test.csv")
        shutil.copy(self.ingestion_artifact.trained_file_path, validated_train)
        shutil.copy(self.ingestion_artifact.test_file_path, validated_test)

        artifact = DataValidationArtifact(
            validation_status=schema_ok and drift_ok and concept_ok,
            valid_train_file_path=validated_train,
            valid_test_file_path=validated_test,
            drift_report_file_path=self.config.drift_report_file_path
        )
        write_yaml_file(self.config.validation_output_path, artifact.__dict__)
        # logging.info(f"Data Validation Completed: {artifact}")
        return artifact
    
    
    
if __name__ == "__main__":
    
    data_ingestion_config = DataIngestionConfig()
    data_ingestion_artifact = DataIngestionArtifact(
        raw_file_path=data_ingestion_config.feature_store_file_path,
        trained_file_path=data_ingestion_config.training_file_path,
        test_file_path=data_ingestion_config.testing_file_path
    )
    data_validation_config = DataValidationConfig()
    data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
    data_validation.initiate_data_validation()