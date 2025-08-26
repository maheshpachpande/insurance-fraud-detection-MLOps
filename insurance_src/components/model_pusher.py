import sys

from insurance_src.cloud_storage.aws_storage import SimpleStorageService
from insurance_src.exceptions import CustomException
from insurance_src.logger import logging
from insurance_src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from insurance_src.entity.config_entity import ModelPusherConfig, ModelEvaluationConfig
from insurance_src.entity.s3_estimator import S3_InsuranceEstimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.usvisa_estimator = S3_InsuranceEstimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=model_pusher_config.s3_model_key_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            logging.info("Uploading artifacts folder to s3 bucket")

            self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)


            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e)
        
        
if __name__ == "__main__":
    try:
        
        from insurance_src.utils.main_utils import read_yaml_file
        
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