import sys
import logging
from insurance_src.components.model_trainer import ModelTrainer
from insurance_src.entity.config_entity import ModelTrainerConfig, DataTransformationConfig
from insurance_src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from insurance_src.utils.main_utils import read_yaml_file


STAGE_NAME = "Model Trainer stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    
    def run(self):
        try:
            
            data_transformation_config = DataTransformationConfig()
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=data_transformation_config.transformed_test_file_path
            )

            model_trainer_config = ModelTrainerConfig()

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_trainer_config
            )
            
            artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed. Artifact: {artifact}")
        except Exception as e:
            logging.error(f"Error in model training stage: {e}")
            sys.exit(1)

if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.run()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        sys.exit(str(e))