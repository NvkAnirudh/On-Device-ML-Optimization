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
            df = pd.read_csv('workbooks/data/friends.csv', encoding='ISO-8859-1')

            logging.info('Read the required data as a pandas dataframe')

            # Dropping duplicates
            df.drop_duplicates(inplace=True)

            logging.info('Cleaned the data')

            # creating the necessary directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Converting df to csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Initiating Train-Test split')
            train_set, remaining_set = train_test_split(df, train_size=0.7, random_state=42)
            valid_set, test_set = train_test_split(remaining_set, test_size=0.15, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            valid_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=='__main__':
    di = DataIngestion()
    train_path, validation_path = di.initiate_data_ingestion()