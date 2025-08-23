# import yaml
# import tempfile
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from typing import Any, Dict, Optional, Type, Union

# from sklearn.model_selection import GridSearchCV
# from sklearn.base import BaseEstimator

# import mlflow
# from mlflow import sklearn as sk

# from insurance_src.logger import logging
# from insurance_src.exceptions import CustomException
# import dagshub

# # ---------------------------
# # 0. Initialize DagsHub tracking (done once)
# # ---------------------------
# dagshub.init(
#     repo_owner="maheshpachpande",
#     repo_name="insuranceFraudDetaction_mlflow",
#     mlflow=True
# )
# mlflow.set_experiment("insurance-fraud-experiment")

# # ---------------------------
# # 1. Config Entities (SRP)
# # ---------------------------
# @dataclass
# class BestModelDetail:
#     best_model: Any
#     best_score: float
#     best_parameters: dict

# # ---------------------------
# # 2. Experiment Logger
# # ---------------------------
# class ExperimentLogger:
#     def __init__(self, experiment_name: str = "insurance-fraud-experiment"):
#         self.experiment_name = experiment_name
#         mlflow.set_experiment(experiment_name)

#     def start_run(self, run_name: str = "run"):
#         """Start MLflow run as context manager"""
#         return mlflow.start_run(run_name=run_name)

#     def log_params(self, params: dict):
#         """Log parameters to MLflow"""
#         for key, value in params.items():
#             mlflow.log_param(key, value)

#     def log_metrics(self, metrics: dict, step: int = 0):
#         """Log metrics to MLflow"""
#         for key, value in metrics.items():
#             mlflow.log_metric(key, value, step=step)

#     def log_artifact(self, file_path: str):
#         """Log an artifact (file) to MLflow"""
#         mlflow.log_artifact(file_path)

#     def log_model(self, model: BaseEstimator, name: str = "model"):
#         """Log a sklearn model to MLflow"""
#         sk.log_model(model, name=name)

# # ---------------------------
# # 3. MLflow Logger Implementation
# # ---------------------------
# class MLflowLogger(ExperimentLogger):
#     """Optional specialized logger (inherits from ExperimentLogger)"""
#     def __init__(self, repo_owner: str, repo_name: str, experiment_name: str):
#         dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
#         mlflow.set_experiment(experiment_name)


# # ---------------------------
# # 4. Model Importer (SRP - separate class to handle dynamic import)
# # ---------------------------
# class ModelImporter:
#     """Handles importing model classes dynamically."""

#     @staticmethod
#     def import_class(full_class_string: str) -> Type[BaseEstimator]:
#         try:
#             module_path, class_name = full_class_string.rsplit(".", 1)
#             module = __import__(module_path, fromlist=[class_name])
#             return getattr(module, class_name)
#         except Exception as e:
#             raise CustomException(f"Error importing {full_class_string}: {e}")


# # ---------------------------
# # 5. Scoring Strategy (OCP - extend scoring rules easily)
# # ---------------------------
# class ScoringStrategy:
#     """Select scoring function based on problem type."""

#     @staticmethod
#     def get_scoring(y: pd.Series) -> str:
#         if pd.api.types.is_numeric_dtype(y):
#             return "r2"  # regression
#         return "accuracy"  # classification


# # ---------------------------
# # 6. Model Factory (SRP - orchestrates search, DIP - uses ExperimentLogger)
# # ---------------------------
# class ModelFactory:
#     def __init__(self, model_config_path: str, logger: ExperimentLogger):
#         self.model_config_path = model_config_path
#         self.logger = logger
#         self.models_config = self._load_model_config()

#     def _load_model_config(self) -> Dict:
#         try:
#             with open(self.model_config_path, "r") as file:
#                 return yaml.safe_load(file)
#         except Exception as e:
#             raise CustomException(e)

#     def get_best_model(
#         self, X_train: Union[pd.DataFrame, np.ndarray], y_train: pd.Series, base_accuracy: float = 0.6
#     ) -> BestModelDetail:
#         scoring = ScoringStrategy.get_scoring(y_train)

#         best_model: Optional[BaseEstimator] = None
#         best_score = -np.inf
#         best_params: dict = {}

#         for model_name, model_info in self.models_config.items():
#             with self.logger.start_run(run_name=model_name):
#                 try:
#                     # Import model
#                     model_class = ModelImporter.import_class(model_info["class"])
#                     param_grid = model_info.get("params", {})

#                     # Log initial params
#                     self.logger.log_params({"model_class": model_info["class"], **param_grid})

#                     # Run GridSearch
#                     search = GridSearchCV(
#                         estimator=model_class(),
#                         param_grid=param_grid,
#                         scoring=scoring,
#                         cv=5,
#                         n_jobs=-1,
#                     )
#                     search.fit(X_train, y_train)

#                     # Log results
#                     self.logger.log_metrics({"cv_best_score": search.best_score_})

#                     # Save YAML config as artifact
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config:
#                         with open(self.model_config_path, "r") as src:
#                             temp_config.write(src.read().encode())
#                         self.logger.log_artifact(temp_config.name)

#                     # Log best model
#                     try:
#                         self.logger.log_model(search.best_estimator_, name="best_model")
#                     except Exception as e:
#                         logging.warning(f"Logging best model failed: {e}")

#                     # Track best model
#                     if search.best_score_ > best_score and search.best_score_ >= base_accuracy:
#                         best_model = search.best_estimator_
#                         best_score = search.best_score_
#                         best_params = search.best_params_

#                     print(f"✅ {model_name} | Score: {search.best_score_:.4f} | Params: {search.best_params_}")

#                 except Exception as e:
#                     raise CustomException(e)

#         if best_model is None:
#             raise ValueError("No model met the base accuracy requirement.")

#         return BestModelDetail(best_model, best_score, best_params)


import yaml
import tempfile
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import joblib
import os

import mlflow
from mlflow import sklearn as sk

from insurance_src.logger import logging
from insurance_src.exceptions import CustomException
import dagshub

# ---------------------------
# 0. Initialize DagsHub tracking (once)
# ---------------------------
dagshub.init(
    repo_owner="maheshpachpande",
    repo_name="insuranceFraudDetaction_mlflow",
    mlflow=True
)
mlflow.set_experiment("insurance-fraud-experiment")

# ---------------------------
# 1. Config Entity
# ---------------------------
@dataclass
class BestModelDetail:
    best_model: Any
    best_score: float
    best_parameters: dict
    best_model_name: str 


# ---------------------------
# 2. Experiment Logger (robust for Dagshub)
# ---------------------------
class ExperimentLogger:
    def __init__(self, experiment_name: str = "insurance-fraud-experiment"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str = "run"):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict, step: int = 0):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(self, file_path: str):
        """Always works on Dagshub: log file artifact"""
        mlflow.log_artifact(file_path)

    def log_model(self, model: BaseEstimator, name: str = "model"):
        """Safe model logging: save locally and log as artifact"""
        try:
            tmp_dir = "artifact/models"
            os.makedirs(tmp_dir, exist_ok=True)
            file_path = os.path.join(tmp_dir, f"{name}.pkl")
            joblib.dump(model, file_path)
            self.log_artifact(file_path)
            logging.info(f"Model saved and logged as artifact: {file_path}")
        except Exception as e:
            logging.warning(f"Logging model failed: {e}")

# ---------------------------
# 3. Model Importer
# ---------------------------
class ModelImporter:
    @staticmethod
    def import_class(full_class_string: str) -> Type[BaseEstimator]:
        try:
            module_path, class_name = full_class_string.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as e:
            raise CustomException(f"Error importing {full_class_string}: {e}")

# ---------------------------
# 4. Scoring Strategy
# ---------------------------
class ScoringStrategy:
    @staticmethod
    def get_scoring(y: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(y):
            return "r2"  # regression
        return "accuracy"  # classification

# ---------------------------
# 5. Model Factory
# ---------------------------
class ModelFactory:
    def __init__(self, model_config_path: str, logger: ExperimentLogger):
        self.model_config_path = model_config_path
        self.logger = logger
        self.models_config = self._load_model_config()

    def _load_model_config(self) -> Dict:
        try:
            with open(self.model_config_path, "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise CustomException(e)

    def get_best_model(
        self, X_train: Union[pd.DataFrame, np.ndarray], y_train: pd.Series, base_accuracy: float = 0.6
    ) -> BestModelDetail:

        scoring = ScoringStrategy.get_scoring(y_train)
        best_model: Optional[BaseEstimator] = None
        best_score = -np.inf
        best_params: dict = {}
        best_model_name: str = ""

        for model_name, model_info in self.models_config.items():
            with self.logger.start_run(run_name=model_name):
                try:
                    # Import model dynamically
                    model_class = ModelImporter.import_class(model_info["class"])
                    param_grid = model_info.get("params", {})

                    # Log initial params
                    self.logger.log_params({"model_class": model_info["class"], **param_grid})

                    # Run GridSearch
                    search = GridSearchCV(
                        estimator=model_class(),
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=5,
                        n_jobs=-1
                    )
                    search.fit(X_train, y_train)

                    # Log results
                    self.logger.log_metrics({"cv_best_score": search.best_score_})

                    # Save YAML config as artifact
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_config:
                        with open(self.model_config_path, "r") as src:
                            temp_config.write(src.read().encode())
                        self.logger.log_artifact(temp_config.name)

                    # Log best model safely
                    self.logger.log_model(search.best_estimator_, name=f"{model_name}_best")

                    # Track best model
                    if search.best_score_ > best_score and search.best_score_ >= base_accuracy:
                        best_model = search.best_estimator_
                        best_score = search.best_score_
                        best_params = search.best_params_
                        best_model_name = model_name  # <- save name

                    print(f"✅ {model_name} | Score: {search.best_score_:.4f} | Params: {search.best_params_}")

                except Exception as e:
                    raise CustomException(e)

        if best_model is None:
            raise ValueError("No model met the base accuracy requirement.")

        return BestModelDetail(
            best_model=best_model,
            best_score=best_score,
            best_parameters=best_params,
            best_model_name=best_model_name 
        )
