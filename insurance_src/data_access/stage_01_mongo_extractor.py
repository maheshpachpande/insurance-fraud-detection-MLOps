
import pandas as pd
import numpy as np
from typing import Optional
from insurance_src.configuration.mondoDB_connection import MongoDBClient
from insurance_src.exceptions import CustomException
import sys

from abc import ABC, abstractmethod




class IDataExtractor(ABC):
    """Interface for all data extractors."""
    
    @abstractmethod
    def export_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        pass


class MongoDataExtractor(IDataExtractor):
    """Extracts data from MongoDB collections."""
    
    def __init__(self, mongo_client: MongoDBClient):
        self.mongo_client = mongo_client

    def export_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            db = self.mongo_client.get_database(database_name)
            collection = db[collection_name]
            
            df = pd.DataFrame(list(collection.find()))
            return df
        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    from insurance_src.constants import COLLECTION_NAME
    from insurance_src.configuration.mondoDB_connection import MongoDBClient, MongoDBConfig
    mongo_client = MongoDBClient(MongoDBConfig())
    extractor = MongoDataExtractor(mongo_client)
    df = extractor.export_as_dataframe(COLLECTION_NAME)
    print(df.head())