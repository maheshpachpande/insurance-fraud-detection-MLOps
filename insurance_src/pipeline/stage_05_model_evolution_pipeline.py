import sys
import logging
from insurance_src.components.model_evaluation import ModelEvaluation
from insurance_src.entity.config_entity import ModelTrainerConfig, DataIngestionConfig, ModelEvaluationConfig
from insurance_src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from insurance_src.utils.main_utils import read_yaml_file


STAGE_NAME = "Model Evaluation stage"

class ModelEvolutionTrainingPipeline:
    def __init__(self):
        pass
    
    
    def run(self):
        try:            
            # Load metrics from model trainer
            metric_path = "artifact/model_trainer/trained_model/metrics.yaml"
            metrics = read_yaml_file(metric_path)

            # Prepare artifacts and configs
            data_ingestion_config = DataIngestionConfig()
            data_ingestion_artifact = DataIngestionArtifact(
                raw_file_path=data_ingestion_config.feature_store_file_path,
                trained_file_path=data_ingestion_config.training_file_path,
                test_file_path=data_ingestion_config.testing_file_path
            )

            model_trainer_config = ModelTrainerConfig()
            test_metric_artifact = ClassificationMetricArtifact(
                f1_score=metrics.get("f1_score", 0.0),
                precision_score=metrics.get("precision_score", 0.0),
                recall_score=metrics.get("recall_score", 0.0)
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_trainer_config.trained_model_file_path,
                test_metric_artifact=test_metric_artifact
            )

            model_eval_config = ModelEvaluationConfig()

            # Run evaluation
            evaluator = ModelEvaluation(
                model_eval_config=model_eval_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            evaluation_artifact = evaluator.run()

        except Exception as e:
            logging.error(f"Error in model training stage: {e}")
            sys.exit(1)

if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvolutionTrainingPipeline()
        obj.run()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        sys.exit(str(e))