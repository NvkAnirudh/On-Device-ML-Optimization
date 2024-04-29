import os
import sys
import pandas as pd
from convokit import Corpus, download
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')
    validation_data_path = os.path.join('artifacts','validation.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Started Data Ingestion Component')

        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)