from pandas import DataFrame
import os
from insurance_src.logger import logging
from abc import ABC, abstractmethod



class IDataSaver(ABC):
    @abstractmethod
    def save(self, data: DataFrame, file_path: str) -> None:
        """Saves DataFrame to a storage location."""
        pass


class CSVDataSaver(IDataSaver):
    def save(self, data: DataFrame, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False, header=True)
        logging.info(f"Data saved to: {file_path}")



