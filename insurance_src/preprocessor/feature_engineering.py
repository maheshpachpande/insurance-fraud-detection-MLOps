

import pandas as pd
import numpy as np
from typing import Optional, List, Union

from insurance_src.logger import logging
from insurance_src.exceptions import CustomException
from insurance_src.utils.main_utils import read_yaml_file
from insurance_src.constants import FEATURE_ENGINEERING_CONFIG_FILE

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer




class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for insurance fraud detection.

    This transformer:
    - Creates categorical features from numeric variables.
    - Drops specified unnecessary columns.
    - Ensures robust handling of missing or unexpected values.

    Attributes
    ----------
    drop_columns : list
        List of columns to drop after transformation.
    """

    def __init__(self, file_path: str = FEATURE_ENGINEERING_CONFIG_FILE): 
        """
        Initialize the transformer.

        Parameters
        ----------
        file_path : str, optional
            Path to the YAML configuration file.
        """
        self.config = read_yaml_file(file_path)
        self.drop_columns = self.config["drop_columns"]
        

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Fit method (no fitting needed for this transformer).

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series or np.ndarray, optional
            Target variable (ignored).

        Returns
        -------
        self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by creating new categorical features and dropping specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Transformed dataset.
        """
        logging.info("Starting feature engineering transformation.")

        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        X_transformed = X.copy()

        # Create categorical features
        X_transformed['policy_deductable_cat'] = np.select(
            [X_transformed['policy_deductable'] == 500,
             X_transformed['policy_deductable'] == 1000,
             X_transformed['policy_deductable'] == 2000],
            ['Low', 'Medium', 'High'], default='Unknown'
        )

        X_transformed['umbrella_limit_cat'] = np.select(
            [X_transformed['umbrella_limit'] == 0,
             X_transformed['umbrella_limit'] == 600000,
             X_transformed['umbrella_limit'] == 1200000,
             X_transformed['umbrella_limit'] == 2000000,
             X_transformed['umbrella_limit'] == 3000000],
            ['None', 'Basic', 'Standard', 'Extended', 'Premium'], default='Unknown'
        )

        X_transformed['bodily_injuries_cat'] = np.select(
            [X_transformed['bodily_injuries'] == 0,
             X_transformed['bodily_injuries'] == 1,
             X_transformed['bodily_injuries'] == 2],
            ['None', 'Minor', 'Major'], default='Unknown'
        )

        X_transformed['incident_hour_cat'] = np.select(
            [(X_transformed['incident_hour_of_the_day'] >= 0) & (X_transformed['incident_hour_of_the_day'] < 6),
             (X_transformed['incident_hour_of_the_day'] >= 6) & (X_transformed['incident_hour_of_the_day'] < 12),
             (X_transformed['incident_hour_of_the_day'] >= 12) & (X_transformed['incident_hour_of_the_day'] < 18),
             (X_transformed['incident_hour_of_the_day'] >= 18) & (X_transformed['incident_hour_of_the_day'] <= 23)],
            ['Early Morning', 'Morning', 'Afternoon', 'Evening'], default='Unknown'
        )

        X_transformed['vehicles_involved_cat'] = np.select(
            [X_transformed['number_of_vehicles_involved'] == 1,
             X_transformed['number_of_vehicles_involved'] == 2,
             X_transformed['number_of_vehicles_involved'] >= 3],
            ['Single Vehicle', 'Two Vehicles', 'Multi-Vehicle'], default='Unknown'
        )

        X_transformed['witnesses_cat'] = np.select(
            [X_transformed['witnesses'] == 0,
             X_transformed['witnesses'] == 1,
             X_transformed['witnesses'] >= 2],
            ['No Witness', 'Single Witness', 'Multiple Witnesses'], default='Unknown'
        )

        X_transformed['policy_csl'] = X_transformed['policy_csl'].astype(str)
        X_transformed['policy_csl_cat'] = np.select(
            [X_transformed['policy_csl'] == '100/300',
             X_transformed['policy_csl'] == '250/500',
             X_transformed['policy_csl'] == '500/1000'],
            ['Basic', 'Standard', 'Premium'], default='Other'
        )

        # Drop columns
        X_transformed.drop(columns=self.drop_columns, inplace=True, errors='ignore')

        logging.info("Feature engineering transformation completed.")
        return X_transformed




# -------------------------------------------
# Data Transformer Class
# -------------------------------------------
class DataTransformer:
    """
    Builds a complete preprocessing pipeline with feature engineering,
    numeric scaling, categorical encoding, and missing value imputation.
    """

    @classmethod
    def get_data_transformer_object(cls, df: pd.DataFrame) -> Pipeline:
        try:
            

            # Feature Engineering Transformer
            feature_engineer = CustomFeatureEngineer()
            df_transformed = feature_engineer.transform(df)

            # Identify feature types
            numeric_features: List[str] = df_transformed.select_dtypes(include=['int64', 'float64']).columns.tolist()
            onehot_features: List[str] = df_transformed.select_dtypes(include=['object']).columns.tolist()
            ordinal_features: List[str] = [
                'policy_deductable_cat', 'umbrella_limit_cat', 'bodily_injuries_cat',
                'incident_hour_cat', 'vehicles_involved_cat', 'witnesses_cat', 'policy_csl_cat'
            ]

            # Ensure ordinal features are in object dtype
            df_transformed[ordinal_features] = df_transformed[ordinal_features].astype(str)

            # Remove ordinal features from one-hot list
            onehot_features = [f for f in onehot_features if f not in ordinal_features]

            # Define ordinal category mapping
            ordinal_mapping = [
                ['Low', 'Medium', 'High', 'Unknown'],
                ['None', 'Basic', 'Standard', 'Extended', 'Premium', 'Unknown'],
                ['None', 'Minor', 'Major', 'Unknown'],
                ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Unknown'],
                ['Single Vehicle', 'Two Vehicles', 'Multi-Vehicle', 'Unknown'],
                ['No Witness', 'Single Witness', 'Multiple Witnesses', 'Unknown'],
                ['Basic', 'Standard', 'Premium', 'Other', 'Unknown']
            ]

            # Pipelines
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            onehot_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore'))
            ])

            ordinal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(categories=ordinal_mapping, handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            # Column Transformer
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', onehot_transformer, onehot_features),
                ('ord', ordinal_transformer, ordinal_features)
            ])

            # Full Pipeline
            full_pipeline = Pipeline(steps=[
                ('feature_engineering', feature_engineer),
                ('preprocessing', preprocessor)
            ])

            logging.info("Data transformer pipeline created successfully.")
            return full_pipeline

        except Exception as e:
            logging.error(f"Error creating data transformer: {str(e)}")
            raise CustomException(e)