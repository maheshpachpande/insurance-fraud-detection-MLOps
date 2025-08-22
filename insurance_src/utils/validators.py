
from typing import Dict
import pandas as pd

from insurance_src.logger import logging


class SchemaValidator:
    """
    Validates whether a pandas DataFrame matches a given schema definition.
    Schema must define column names, and optionally which are numerical/categorical.
    """

    def __init__(self, schema_definition: Dict):
        """
        :param schema_definition: Dict specifying 'columns', 'numerical_columns', 'categorical_columns'
        :param logger: Optional, custom logger instance
        """
        self.schema = schema_definition

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Run all schema validations.
        """
        columns_valid = self._validate_columns(df)
        numeric_valid = self._validate_numerical_columns(df)
        categorical_valid = self._validate_categorical_columns(df)
        return columns_valid and numeric_valid and categorical_valid

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        expected_columns = set(self.schema.get("columns", {}).keys())
        actual_columns = set(df.columns)
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            logging.warning("Missing columns: %s", missing_columns)
        return not missing_columns

    def _validate_numerical_columns(self, df: pd.DataFrame) -> bool:
        required_numerical = set(self.schema.get("numerical_columns", []))
        missing_numerical = required_numerical - set(df.columns)
        if missing_numerical:
            logging.warning("Missing numerical columns: %s", missing_numerical)
        return not missing_numerical

    def _validate_categorical_columns(self, df: pd.DataFrame) -> bool:
        required_categorical = set(self.schema.get("categorical_columns", []))
        missing_categorical = required_categorical - set(df.columns)
        if missing_categorical:
            logging.warning("Missing categorical columns: %s", missing_categorical)
        return not missing_categorical



if __name__ == "__main__":
    from insurance_src.utils.main_utils import read_yaml_file
    from insurance_src.constants import SCHEMA_FILE_PATH
    
    df = pd.read_csv("artifact/data_ingestion/ingested/train.csv")
    # df = pd.read_csv("artifact/data_ingestion/feature_store/raw.csv")
    schema = read_yaml_file(SCHEMA_FILE_PATH)
    validator = SchemaValidator(schema)
    assert validator.validate(df)

