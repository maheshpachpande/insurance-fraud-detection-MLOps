# src/services/insurance_data_service.py
import pandas as pd
from typing import Optional

from insurance_src.data_access.stage_01_mongo_extractor import IDataExtractor
from insurance_src.data_access.stage_02_data_cleaner import DataCleaner




class InsuranceDataService:
    """Service layer for fetching and cleaning insurance data."""
    
    def __init__(self, extractor: IDataExtractor, cleaner: DataCleaner):
        self.extractor = extractor
        self.cleaner = cleaner
    
    def get_clean_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        df = self.extractor.export_as_dataframe(collection_name, database_name)       
        df = self.cleaner.remove_mongo_id(df)        
        df = self.cleaner.replace_na_with_nan(df)
        df = self.cleaner.drop_duplicates(df)
        return df


if __name__ == "__main__":
    from insurance_src.constants import COLLECTION_NAME
    from insurance_src.configuration.mondoDB_connection import MongoDBClient, MongoDBConfig
    from insurance_src.data_access.stage_01_mongo_extractor import MongoDataExtractor
    
    mongo_client = MongoDBClient(MongoDBConfig())
    extractor = MongoDataExtractor(mongo_client)
    cleaner = DataCleaner()
    service = InsuranceDataService(extractor, cleaner)
    df = service.get_clean_dataframe(COLLECTION_NAME)
    print(df.head())