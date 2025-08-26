import sys
import logging
from insurance_src.exceptions import CustomException
from insurance_src.components.model_pusher import ModelPusher
from insurance_src.entity.config_entity import ModelPusherConfig, ModelEvaluationConfig
from insurance_src.entity.artifact_entity import ModelEvaluationArtifact
from insurance_src.utils.main_utils import read_yaml_file


STAGE_NAME = "Model Pusher stage"



class ModelPusherTrainingPipeline:
    def __init__(self):
        pass
    
    
    def run(self):
        try:            
            art = read_yaml_file("artifact/model_evaluation/model_evaluation_artifact.yaml")
            
            model_eval_config = ModelEvaluationConfig()
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=art.is_model_accepted,
                trained_model_path=art.trained_model_path,
                s3_model_path=model_eval_config.s3_model_key_path,
                changed_accuracy=art.changed_accuracy
            )
            model_pusher_config = ModelPusherConfig()
            model_pusher = ModelPusher(model_pusher_config=model_pusher_config
                                    , model_evaluation_artifact=model_evaluation_artifact)
            
            model_pusher.initiate_model_pusher()
        except Exception as e:
            raise CustomException(e)

if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPusherTrainingPipeline()
        obj.run()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e)