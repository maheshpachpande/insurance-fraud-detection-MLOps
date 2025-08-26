from typing import Optional

from insurance_src.exceptions import CustomException
from insurance_src.logger import logging

from insurance_src.components.data_ingestion import DataIngestion
from insurance_src.components.data_validation import DataValidation
from insurance_src.components.data_transformation import DataTransformation
from insurance_src.components.model_trainer import ModelTrainer
from insurance_src.components.model_evaluation import ModelEvaluation
from insurance_src.components.model_pusher import ModelPusher

from insurance_src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)

from insurance_src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)


class TrainPipeline:
    """
    Orchestrates the complete ML pipeline: ingestion → validation → transformation
    → model training → evaluation → pushing.
    """

    def __init__(
        self,
        ingestion_config: Optional[DataIngestionConfig] = None,
        validation_config: Optional[DataValidationConfig] = None,
        transformation_config: Optional[DataTransformationConfig] = None,
        trainer_config: Optional[ModelTrainerConfig] = None,
        evaluation_config: Optional[ModelEvaluationConfig] = None,
        pusher_config: Optional[ModelPusherConfig] = None,
    ) -> None:
        try:
            self.data_ingestion_config = ingestion_config or DataIngestionConfig()
            self.data_validation_config = validation_config or DataValidationConfig()
            self.data_transformation_config = transformation_config or DataTransformationConfig()
            self.model_trainer_config = trainer_config or ModelTrainerConfig()
            self.model_evaluation_config = evaluation_config or ModelEvaluationConfig()
            self.model_pusher_config = pusher_config or ModelPusherConfig()
        except Exception as e:
            raise CustomException(e)

    # ------------------- Stage 1 -------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("▶️ Starting data ingestion...")
            ingestion = DataIngestion(self.data_ingestion_config)
            artifact = ingestion.initiate_data_ingestion()
            logging.info(f"✅ Data ingestion completed: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e)

    # ------------------- Stage 2 -------------------
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("▶️ Starting data validation...")
            validator = DataValidation(ingestion_artifact=data_ingestion_artifact, validation_config=self.data_validation_config)
            artifact = validator.initiate_data_validation()
            logging.info(f"✅ Data validation completed: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e)

    # ------------------- Stage 3 -------------------
    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("▶️ Starting data transformation...")
            transformer = DataTransformation(
                validation_artifact=data_validation_artifact,
                transformation_config=self.data_transformation_config
            )
            artifact = transformer.initiate_transformation()
            logging.info(f"✅ Data transformation completed: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e)

    # ------------------- Stage 4 -------------------
    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("▶️ Starting model training...")
            trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )
            artifact = trainer.initiate_model_trainer()
            logging.info(f"✅ Model training completed: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e)

    # ------------------- Stage 5 -------------------
    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("▶️ Starting model evaluation...")
            evaluator = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            artifact = evaluator.run()
            logging.info(f"✅ Model evaluation completed: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e)

    # ------------------- Stage 6 -------------------
    def start_model_pusher(
        self, model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:
        try:
            logging.info("▶️ Starting model pushing...")
            pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_evaluation_artifact=model_evaluation_artifact
            )
            artifact = pusher.initiate_model_pusher()
            logging.info(f"✅ Model pushing completed: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e)

    # ------------------- Run Pipeline -------------------
    def run_pipeline(self) -> None:
        try:
            logging.info("🚀 ML Training Pipeline started")

            ingestion_artifact = self.start_data_ingestion()
            validation_artifact = self.start_data_validation(ingestion_artifact)
            transformation_artifact = self.start_data_transformation(validation_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            evaluation_artifact = self.start_model_evaluation(
                ingestion_artifact, trainer_artifact
            )

            if not evaluation_artifact.is_model_accepted:
                logging.info("❌ Model not accepted. Stopping pipeline.")
                return

            self.start_model_pusher(evaluation_artifact)
            logging.info("🏁 ML Training Pipeline completed successfully")
        except Exception as e:
            raise CustomException(e)
