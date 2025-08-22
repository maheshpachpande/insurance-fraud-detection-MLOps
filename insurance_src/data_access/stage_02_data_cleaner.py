# src/data_access/data_cleaner.py
import pandas as pd
import numpy as np

class DataCleaner:
    """Cleans extracted DataFrames."""
    
    @staticmethod
    def remove_mongo_id(df: pd.DataFrame) -> pd.DataFrame:
        if "_id" in df.columns:
            df = df.drop(columns=["_id"], axis=1)
        return df
    
    @staticmethod
    def replace_na_with_nan(df: pd.DataFrame) -> pd.DataFrame:
        return df.replace({"na": np.nan})
    
    @staticmethod
    def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()