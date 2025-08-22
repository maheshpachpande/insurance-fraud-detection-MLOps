from insurance_src.logger import logging
from insurance_src.exceptions import CustomException
from insurance_src.components.data_transformation import DataTransformation
from insurance_src.entity.config_entity import DataTransformationConfig, DataValidationConfig
from insurance_src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from insurance_src.utils.main_utils import read_yaml_file



STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass
    
    
    def run(self):
        try:        
            data_validation_config = DataValidationConfig()
            val_artifact_yaml = read_yaml_file(data_validation_config.validation_output_path)
            
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=val_artifact_yaml["validation_status"],
                valid_train_file_path=data_validation_config.validated_train_file_path,
                valid_test_file_path=data_validation_config.validated_test_file_path,
                drift_report_file_path=data_validation_config.drift_report_file_path
            )
                
            data_transformation_config = DataTransformationConfig()
            data_transformation = DataTransformation(
                validation_artifact=data_validation_artifact,
                transformation_config=data_transformation_config
            )
            data_transformation.initiate_transformation()
        except Exception as e:
            logging.error(f"Error in data transformation pipeline: {e}")
            raise CustomException(e)
        
        
if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.run()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e)