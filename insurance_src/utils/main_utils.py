import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame
from box import ConfigBox
from insurance_src.exceptions import CustomException
from insurance_src.logger import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from src.entity.artifact_entity import ClassificationMetricArtifact




# 

def read_yaml_file(file_path: str) -> ConfigBox:
    try:
        with open(file_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not isinstance(content, dict):
                raise ValueError(f"YAML file {file_path} must contain a mapping at the root.")
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e)




# def evaluate_classification_metrics(y_true, y_pred) -> ClassificationMetricArtifact:
#     """
#     Calculates classification metrics and returns them in a ClassificationMetricArtifact object.

#     Args:
#         y_true (array-like): Ground truth (correct) labels.
#         y_pred (array-like): Predicted labels by the model.

#     Returns:
#         ClassificationMetricArtifact: Contains f1_score, precision_score, and recall_score.
#     """
#     accuracy = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)

#     # Log metrics for debugging (optional)
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")

#     return ClassificationMetricArtifact(
#         f1_score=f1,
#         precision_score=precision,
#         recall_score=recall
#     )




def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e ) 


from typing import TypeVar, Type
import dill
import sys

T = TypeVar("T")

def load_object(file_path: str, expected_type: Type[T]) -> T:
    """
    Load an object from file and ensure it matches the expected type.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        if not isinstance(obj, expected_type):
            raise TypeError(f"Expected {expected_type}, got {type(obj)}")
        return obj
    except Exception as e:
        raise CustomException(e )


def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e ) 


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise CustomException(e ) 


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise CustomException(e ) 


# def drop_columns(df: DataFrame, cols: list)-> DataFrame:

#     """
#     drop the columns form a pandas DataFrame
#     df: pandas DataFrame
#     cols: list of columns to be dropped
#     """
#     logging.info("Entered drop_columns methon of utils")

#     try:
#         df = df.drop(columns=cols, axis=1)

#         logging.info("Exited the drop_columns method of utils")
        
#         return df
#     except Exception as e:
#         raise CustomException(e ) 