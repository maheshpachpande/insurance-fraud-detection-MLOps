import os
import sys
import yaml
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from insurance_src.exceptions import CustomException
from insurance_src.logger import logging
from insurance_src.utils.main_utils import load_numpy_array_data, load_object, save_object
from insurance_src.entity.model_factory import ModelFactory, BestModelDetail, ExperimentLogger
from insurance_src.entity.config_entity import ModelTrainerConfig, DataTransformationConfig
from insurance_src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from insurance_src.entity.model_wrapper import InsuranceModel


class ModelTrainer:
    """
    Handles training, evaluation, and persistence of machine learning models.

    Responsibilities:
    - Load transformed datasets and preprocessing objects.
    - Select and train the best model using ModelFactory.
    - Evaluate model using classification metrics.
    - Detect overfitting based on threshold.
    - Save trained model and metrics for downstream use.
    """

    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_best_model_and_metrics(
        self, train_array: np.ndarray, test_array: np.ndarray
    ) -> Tuple[BestModelDetail, ClassificationMetricArtifact]:
        """
        Train multiple models and return the best model along with evaluation metrics.

        Args:
            train_array (np.ndarray): Training data including target column.
            test_array (np.ndarray): Testing data including target column.

        Returns:
            Tuple[BestModelDetail, ClassificationMetricArtifact]: Best model and its evaluation metrics.

        Raises:
            CustomException: If training or evaluation fails.
            Exception: If overfitting is detected.
        """
        try:
            logging.info("Starting model selection using ModelFactory")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path,
                                         logger=ExperimentLogger())

            # Split features and targets
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Encode labels
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)

            # Select best model
            best_model_detail = model_factory.get_best_model(
                X_train=x_train,
                y_train=pd.Series(y_train),
                base_accuracy=self.model_trainer_config.expected_accuracy
            )

            # Predict and evaluate
            y_train_pred = best_model_detail.best_model.predict(x_train)
            y_test_pred = best_model_detail.best_model.predict(x_test)

            f1_train = f1_score(y_train, y_train_pred)
            f1_test = f1_score(y_test, y_test_pred)

            # Overfitting check
            if (f1_train - f1_test) > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception(
                    f"Overfitting detected: F1 train={f1_train:.3f}, F1 test={f1_test:.3f}"
                )

            metrics = ClassificationMetricArtifact(
                f1_score=float(f1_test),
                precision_score=float(precision_score(y_test, y_test_pred)),
                recall_score=float(recall_score(y_test, y_test_pred))
            )

            return best_model_detail, metrics

        except Exception as e:
            logging.error(f"Error in get_best_model_and_metrics: {e}")
            raise CustomException(e)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training pipeline: loads data, selects best model,
        evaluates, persists trained model and metrics.

        Returns:
            ModelTrainerArtifact: Artifact containing trained model path and metrics.

        Raises:
            CustomException: If any step in the pipeline fails.
        """
        logging.info("Initiating model training process")
        try:
            # Load transformed datasets
            train_array = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_array = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            # Train and evaluate model
            best_model_detail, metrics = self.get_best_model_and_metrics(train_array, test_array)

            # Load preprocessing pipeline
            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path,
                expected_type=Pipeline
            )

            # Check if best model meets expected accuracy
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No model met the expected accuracy threshold")

            # Combine preprocessing and model into InsuranceModel
            insurance_model = InsuranceModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model
            )

            # Save model
            save_object(self.model_trainer_config.trained_model_file_path, insurance_model)
            logging.info(f"Trained model saved at: {self.model_trainer_config.trained_model_file_path}")

            # Save metrics as YAML
            metrics_path = os.path.join(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                "metrics.yaml"
            )
            with open(metrics_path, "w") as f:
                yaml.dump(metrics.__dict__, f)
            logging.info(f"Saved evaluation metrics at: {metrics_path}")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                test_metric_artifact=metrics
            )

        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise CustomException(e)


# ---------------------------
# Example execution (for standalone use)
# ---------------------------
if __name__ == "__main__":
    try:
        data_transformation_config = DataTransformationConfig()
        data_transformation_artifact = DataTransformationArtifact(
            transformed_object_file_path=data_transformation_config.transformed_object_file_path,
            transformed_train_file_path=data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=data_transformation_config.transformed_test_file_path
        )

        model_trainer_config = ModelTrainerConfig()
        trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=model_trainer_config
        )

        artifact = trainer.initiate_model_trainer()
        logging.info(f"Model training completed successfully: {artifact}")

    except Exception as e:
        # logging.error(f"Error in get_best_model_and_metrics", exc_info=True)
        raise CustomException(e)

