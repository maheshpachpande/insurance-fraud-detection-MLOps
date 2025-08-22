from __future__ import annotations


from typing import Optional
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from insurance_src.entity.config_entity import DataIngestionConfig
from insurance_src.entity.artifact_entity import DataIngestionArtifact

from insurance_src.exceptions import CustomException
from insurance_src.logger import logging
from insurance_src.utils.main_utils import read_yaml_file
from insurance_src.constants import SCHEMA_FILE_PATH

from insurance_src.data_access.stage_01_mongo_extractor import MongoDataExtractor
from insurance_src.data_access.stage_02_data_cleaner import DataCleaner
from insurance_src.data_access.stage_03_data_saver import CSVDataSaver
from insurance_src.data_access.insurance_data_service import InsuranceDataService

from insurance_src.configuration.mondoDB_connection import MongoDBClient, MongoDBConfig




class DataIngestion:
    def __init__(
        self,
        config: Optional[DataIngestionConfig] = None,
        service: Optional[InsuranceDataService] = None,
        saver: Optional[CSVDataSaver] = None
    ) -> None:
        try:
            self.config = config or DataIngestionConfig()

            # Ensure service is always set
            if service is None:
                mongo_client = MongoDBClient(MongoDBConfig())
                extractor = MongoDataExtractor(mongo_client)
                cleaner = DataCleaner()
                service = InsuranceDataService(extractor, cleaner)

            self.service: InsuranceDataService = service  
            self.saver: CSVDataSaver = saver or CSVDataSaver()
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomException(e)


    def export_data_into_feature_store(self) -> DataFrame:
        """Fetches and cleans data, then saves it to feature store."""
        try:
            logging.info("Fetching and cleaning data from source...")
            df = self.service.get_clean_dataframe(self.config.collection_name)

            if df.empty:
                raise CustomException("Extracted dataframe is empty." )

            self.saver.save(df, self.config.feature_store_file_path)
            return df
        except Exception as e:
            raise CustomException(e)


    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """Splits data into train and test sets and saves them."""
        try:
            drop_columns = self._schema_config.get("drop_columns", [])
            if drop_columns:
                
                dataframe = dataframe.drop(columns=drop_columns, errors="ignore")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.config.train_test_split_ratio,
                random_state=42
            )

            self.saver.save(train_set, self.config.training_file_path)
            self.saver.save(test_set, self.config.testing_file_path)

            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
        except Exception as e:
            raise CustomException(e)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Main orchestration method."""
        try:
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df)

            artifact = DataIngestionArtifact(
                raw_file_path=self.config.feature_store_file_path,
                trained_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path
            )
            return artifact
        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        artifact = ingestion.initiate_data_ingestion()
        print(artifact)
    except Exception as e:
        logging.error(f"Error in data ingestion pipeline: {e}")
        raise CustomException(e)
