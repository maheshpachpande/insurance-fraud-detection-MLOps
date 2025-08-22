


import pymongo
import certifi # for ssl
from typing import Optional, cast
from abc import ABC, abstractmethod

from pymongo.database import Database  

from insurance_src.exceptions import CustomException
from insurance_src.logger import logging

from insurance_src.entity.config_entity import MongoDBConfig


from dotenv import load_dotenv
# Load environment variables once at app start
load_dotenv()


CA_CERT_PATH = certifi.where()


# ----------------- Interface (DIP + ISP) -----------------
class BaseDatabaseClient(ABC):
    """Abstract base class for any database client."""

    @abstractmethod
    def connect(self) -> None:
        """Establish a database connection."""
        pass

    @abstractmethod
    def get_database(self, name: Optional[str] = None) -> Database:
        """Retrieve a database instance by name."""
        pass


# ----------------- MongoDB Client (OCP, DIP) -----------------
class MongoDBClient(BaseDatabaseClient):
    """
    MongoDB client implementation following SOLID principles.
    Reuses a single connection instance (Singleton pattern).
    """

    _shared_client: Optional[pymongo.MongoClient] = None  

    def __init__(self, config: MongoDBConfig):
        self._config = config
        self._client: Optional[pymongo.MongoClient] = None

    def connect(self) -> None:
        """Establish a MongoDB connection if not already connected."""
        try:
            if MongoDBClient._shared_client is None:
                MongoDBClient._shared_client = pymongo.MongoClient(
                    self._config.uri,
                    tlsCAFile=CA_CERT_PATH
                )
                logging.info("MongoDB connection established successfully.")
            self._client = MongoDBClient._shared_client
        except Exception as e:
            raise CustomException(e)

    def get_database(self, name: Optional[str] = None) -> Database:
        """Retrieve a database by name, defaulting to config default_db."""
        if self._client is None:
            self.connect()
        return cast(pymongo.MongoClient, self._client)[name or self._config.default_db]
    
    
    
if __name__ == "__main__":
    mongo_client = MongoDBClient(MongoDBConfig())
    print(mongo_client.get_database().name)
