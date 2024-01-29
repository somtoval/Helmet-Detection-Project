import os 
import sys
from zipfile import ZipFile
import shutil
from helmet.entity.config_entity import DataIngestionConfig
from helmet.entity.artifacts_entity import DataIngestionArtifacts
from helmet.exception import HelmetException
from helmet.logger import logging
from helmet.constants import *
from pathlib import Path

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def getfile(self):
        try:
            print(f'ORIGINAL LOCATION: {self.data_ingestion_config.ZIP_FILE_NAME}, NEW LOCATION: {self.data_ingestion_config.ZIP_FILE_PATH}')
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            shutil.copy(self.data_ingestion_config.ZIP_FILE_NAME, self.data_ingestion_config.ZIP_FILE_PATH)
            logging.info(f"Data file Gotten")
        except Exception as e:
            raise HelmetException(e, sys) from e

    def unzip_and_clean(self):
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            return self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR, self.data_ingestion_config.VALID_DATA_ARTIFACT_DIR
        except Exception as e:
            raise HelmetException(e, sys) from e
        
    def initiate_data_ingestion(self) -> DataIngestionArtifacts: 
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")
        try:
            self.getfile()
            train_file_path, test_file_path, valid_file_path= self.unzip_and_clean()
            logging.info("Unzipped file and splited into train, test and valid")
            data_ingestion_artifact = DataIngestionArtifacts(train_file_path=train_file_path, 
                                                                test_file_path=test_file_path,
                                                                valid_file_path=valid_file_path)
            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact

        except Exception as e:
            raise HelmetException(e, sys) from e
        
# ex_config = DataIngestionConfig()
        
# example = DataIngestion(ex_config)
# example.initiate_data_ingestion()