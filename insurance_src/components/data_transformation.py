import sys
import numpy as np
import pandas as pd
from typing import Union, Tuple

from imblearn.combine import SMOTETomek

from insurance_src.logger import logging
from insurance_src.exceptions import CustomException
from insurance_src.preprocessor.target_encoder import TargetValueMapping
from insurance_src.preprocessor.feature_engineering import DataTransformer
from insurance_src.constants import TARGET_COLUMN
from insurance_src.utils.main_utils import save_numpy_array_data, save_object, read_yaml_file
from insurance_src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from insurance_src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact


# ------------------ SRP: DataReader ------------------
class DataReader:
    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e)


# ------------------ SRP: Target Mapper ------------------
class TargetMapper:
    def __init__(self):
        self.mapping = TargetValueMapping().to_dict()

    def map_target(self, df: pd.DataFrame) -> pd.Series:
        try:
            return df[TARGET_COLUMN].replace(self.mapping)
        except Exception as e:
            raise CustomException(e)


# ------------------ SRP: Imbalance Handler ------------------from typing import Unionfrom typing import Union, Tuple
class ImbalanceHandler:
    def __init__(self, sampler=None):
        self.sampler = sampler if sampler else SMOTETomek(sampling_strategy="minority")

    def resample(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        try:            
            resampled = self.sampler.fit_resample(X, y)
            X_res = resampled[0]
            y_res = resampled[1]
            
            # Force conversion to NumPy arrays for type safety
            if isinstance(X_res, (pd.DataFrame, pd.Series)):
                X_res = X_res.to_numpy()
            else:
                X_res = np.array(X_res)

            if isinstance(y_res, (pd.DataFrame, pd.Series)):
                y_res = y_res.to_numpy().ravel()  # flatten Series/DataFrame to 1D
            else:
                y_res = np.array(y_res).ravel()

            return X_res, y_res
        except Exception as e:
            raise CustomException(e)


# ------------------ SRP: Data Transformation ------------------
class DataTransformation:
    def __init__(self, validation_artifact: DataValidationArtifact,
                 transformation_config: DataTransformationConfig):
        try:
            self.validation_artifact = validation_artifact
            self.transformation_config = transformation_config
            self.data_transformer = DataTransformer()
            self.target_mapper = TargetMapper()
            self.imbalance_handler = ImbalanceHandler()
        except Exception as e:
            raise CustomException(e)

    def _preprocess_features(self, df: pd.DataFrame):
        X = df.drop(columns=[TARGET_COLUMN], axis=1)
        y = self.target_mapper.map_target(df)
        return X, y

    def _transform_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        preprocessor = self.data_transformer.get_data_transformer_object(X_train)
        preprocessor_obj = preprocessor.fit(X_train)
        X_train_transformed = preprocessor_obj.transform(X_train)
        X_test_transformed = preprocessor_obj.transform(X_test)
        return preprocessor_obj, X_train_transformed, X_test_transformed

    def initiate_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.validation_artifact.validation_status:
                logging.warning("⚠️ Data validation failed. Proceeding anyway (debug mode).")

            logging.info("Reading train and test data.")
            train_df = DataReader.read_csv(self.validation_artifact.valid_train_file_path)
            test_df = DataReader.read_csv(self.validation_artifact.valid_test_file_path)

            logging.info("Preprocessing features and mapping target.")
            X_train, y_train = self._preprocess_features(train_df)
            X_test, y_test = self._preprocess_features(test_df)

            logging.info("Transforming features.")
            preprocessor_obj, X_train_transformed, X_test_transformed = self._transform_features(X_train, X_test)

            logging.info("Handling class imbalance.")
            X_train_final, y_train_final = self.imbalance_handler.resample(X_train_transformed, y_train)
            X_test_final, y_test_final = self.imbalance_handler.resample(X_test_transformed, y_test)

            logging.info("Saving transformed arrays and preprocessor object.")
            train_arr = np.c_[X_train_final, np.array(y_train_final)]
            test_arr = np.c_[X_test_final, np.array(y_test_final)]

            save_numpy_array_data(self.transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.transformation_config.transformed_test_file_path, test_arr)
            save_object(self.transformation_config.transformed_object_file_path, preprocessor_obj)

            return DataTransformationArtifact(
                transformed_object_file_path=self.transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    try:
        
        data_validation_config = DataValidationConfig()
        val_artifact_yaml = read_yaml_file(data_validation_config.validation_output_path)
        
        
        data_validation_artifact = DataValidationArtifact(
            validation_status=val_artifact_yaml["validation_status"],
            valid_train_file_path=data_validation_config.validated_train_file_path,
            valid_test_file_path=data_validation_config.validated_test_file_path,
            drift_report_file_path=data_validation_config.drift_report_file_path
        )
            
        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation(
            validation_artifact=data_validation_artifact,
            transformation_config=data_transformation_config
        )
        data_transformation.initiate_transformation()
    except Exception as e:
        logging.error(f"Error in data transformation pipeline: {e}")
        raise CustomException(e)