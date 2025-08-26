import sys
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from dotenv import load_dotenv

from insurance_src.entity.config_entity import ModelEvaluationConfig, DataIngestionConfig, ModelTrainerConfig
from insurance_src.entity.artifact_entity import (
    ClassificationMetricArtifact,
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)
from insurance_src.entity.s3_estimator import S3_InsuranceEstimator
from insurance_src.preprocessor.target_encoder import TargetValueMapping
from insurance_src.exceptions import CustomException
from insurance_src.constants import TARGET_COLUMN
from insurance_src.logger import logging
from insurance_src.utils.main_utils import read_yaml_file

load_dotenv()


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    """Evaluates a trained model against the production model."""

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def _get_best_model(self) -> Optional[S3_InsuranceEstimator]:
        """Fetch the production model from S3 if it exists."""
        try:
            estimator = S3_InsuranceEstimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=self.model_eval_config.s3_model_key_path
            )
            return estimator if estimator.is_model_present(self.model_eval_config.s3_model_key_path) else None
        except Exception as e:
            raise CustomException(f"Error fetching best model: {e}") from e

    def _prepare_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess test data."""
        try:
            df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x = df.drop(TARGET_COLUMN, axis=1)
            y = df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())
            return x, y
        except Exception as e:
            raise CustomException(f"Error loading test data: {e}") from e

    def evaluate(self) -> EvaluateModelResponse:
        """Compare trained model with production model and return evaluation response."""
        try:
            x, y = self._prepare_data()
            trained_f1 = self.model_trainer_artifact.test_metric_artifact.f1_score

            best_model = self._get_best_model()
            best_f1 = 0.0

            if best_model:
                y_pred = np.array(best_model.predict(x), dtype=int)

                if y.dtype == object or isinstance(y.iloc[0], str):
                    y = LabelEncoder().fit_transform(y)

                best_f1 = float(f1_score(y, y_pred))

            is_accepted = trained_f1 > best_f1
            difference = trained_f1 - best_f1

            response = EvaluateModelResponse(
                trained_model_f1_score=trained_f1,
                best_model_f1_score=best_f1,
                is_model_accepted=is_accepted,
                difference=difference
            )

            logging.info(f"Model evaluation response: {response}")
            return response

        except Exception as e:
            raise CustomException(f"Error during model evaluation: {e}") from e

    def save_artifact(self, response: EvaluateModelResponse) -> ModelEvaluationArtifact:
        """Persist evaluation results as an artifact JSON file."""
        try:
            artifact = ModelEvaluationArtifact(
                is_model_accepted=response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=response.difference
            )

            os.makedirs(os.path.dirname(self.model_eval_config.artifact_file_path), exist_ok=True)
            with open(self.model_eval_config.artifact_file_path, 'w') as f:
                json.dump(asdict(artifact), f, indent=4)

            logging.info(f"Saved model evaluation artifact at {self.model_eval_config.artifact_file_path}")
            return artifact

        except Exception as e:
            raise CustomException(f"Error saving model evaluation artifact: {e}") from e

    def run(self) -> ModelEvaluationArtifact:
        """Full evaluation workflow."""
        response = self.evaluate()
        return self.save_artifact(response)


# ----------------- Example Usage ----------------- #
if __name__ == "__main__":
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
