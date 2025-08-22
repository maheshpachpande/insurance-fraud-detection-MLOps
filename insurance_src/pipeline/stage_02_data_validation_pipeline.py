import sys
from insurance_src.components.data_validation import DataValidation
from insurance_src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from insurance_src.entity.config_entity import DataValidationConfig, DataIngestionConfig
from insurance_src.logger import logging
from insurance_src.exceptions import CustomException


STAGE_NAME = "Data Validatation stage"

class DataValidationPipeline:
    def __init__(self):
        pass
    
        
    def run(self):
        try:
            # Setup Data Ingestion Artifact based on previous step outputs
            data_ingestion_config = DataIngestionConfig()
            data_ingestion_artifact = DataIngestionArtifact(
                raw_file_path=data_ingestion_config.feature_store_file_path,
                trained_file_path=data_ingestion_config.training_file_path,
                test_file_path=data_ingestion_config.testing_file_path
            )

            # Prepare Data Validation Config
            data_validation_config = DataValidationConfig()

            validator = DataValidation(
                ingestion_artifact=data_ingestion_artifact,
                validation_config=data_validation_config
            )

            artifact = validator.initiate_data_validation()
            logging.info(f"Data validation completed. Artifact: {artifact}")

        except Exception as e:
            logging.error(f"Error in data validation stage: {e}")
            sys.exit(1)

if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationPipeline()
        obj.run()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e)