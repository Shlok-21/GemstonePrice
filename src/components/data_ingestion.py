import os
import sys
from logger import logging
from exception import CustomException

from data_transformation import DataTransformation

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionConfig():
    train_data_path = os.path.join('artifact','train.csv')
    test_data_path = os.path.join('artifact','test.csv')
    raw_data_path = os.path.join('artifact','raw_data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Data ingestion has started')
        try:
            df = pd.read_csv('notebooks\cubic_zirconia.csv')
            logging.info('Dataset read successfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            train_data, test_data = train_test_split(df,test_size= 0.2, random_state= 5)

            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Train test split complete')
        
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    di = DataIngestion()
    train_data, test_data = di.initiate_data_ingestion()

    data_transform = DataTransformation()
    data_transform.initiate_data_transform(train_data, test_data)