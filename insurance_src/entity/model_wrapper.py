from typing import Protocol, Union
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from insurance_src.logger import logging


class Predictable(Protocol):
    """
    Protocol for any model with a scikit-learn style `predict` method.
    Ensures type safety and enforces a consistent interface for prediction.
    """
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, list]:
        ...


class InsuranceModel:
    """
    Wrapper class for combining a preprocessing pipeline with a trained model
    into a single production-ready prediction pipeline.

    This class follows SOLID principles:
    - **Single Responsibility**: Encapsulates preprocessing + prediction logic.
    - **Open/Closed**: Easily extendable for new model types or preprocessing.
    - **Liskov Substitution**: Works with any model that follows scikit-learn's predict interface.
    - **Interface Segregation**: Relies on a minimal `Predictable` protocol.
    - **Dependency Inversion**: Depends on abstractions (protocols), not concrete implementations.

    Attributes:
        preprocessing_object (Pipeline): Fitted preprocessing pipeline
        trained_model_object (Predictable): Trained ML model object
    """

    def __init__(self, preprocessing_object: Pipeline, trained_model_object: Predictable) -> None:
        """
        Initialize the InsuranceModel.

        Args:
            preprocessing_object (Pipeline): Pre-fitted preprocessing pipeline
            trained_model_object (Predictable): Pre-trained ML model object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> Union[np.ndarray, dict]:
        """
        Transform raw inputs using the preprocessing pipeline and perform predictions
        with the trained model.

        Args:
            dataframe (pd.DataFrame): Raw input data

        Returns:
            np.ndarray | dict: Predictions from the trained model or error information
        """
        logging.info("Starting prediction in InsuranceModel")

        try:
            logging.info("Applying preprocessing transformations")
            transformed_features = self.preprocessing_object.transform(dataframe)

            logging.info("Performing predictions using trained model")
            predictions = self.trained_model_object.predict(transformed_features)

            logging.info("Prediction successful")
            return predictions

        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            return {"status": False, "error": str(e)}

    def __repr__(self) -> str:
        """
        Developer-friendly representation of the InsuranceModel.
        """
        return (
            f"InsuranceModel(\n"
            f"  Preprocessor={type(self.preprocessing_object).__name__},\n"
            f"  Model={type(self.trained_model_object).__name__}\n)"
        )

    def __str__(self) -> str:
        """
        User-friendly string representation of the InsuranceModel.
        """
        return f"InsuranceModel using {type(self.trained_model_object).__name__}"
