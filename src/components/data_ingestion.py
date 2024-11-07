import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import List
from data_transformation import DataTransforamtion
from model_trainer import ModelTrainer

@dataclass
class _DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = _DataIngestionConfig()
    
    def initiate_data_ingestion(self) -> List[str]:
        '''
        This method will provide the paths for the train and test sets.
        '''
        logging.info("Enter the data ingestion method or component")

        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    transformation = DataTransforamtion()

    train_arr, test_arr, _ = transformation.initiate_data_trainsformation(train_path=train_path, test_path=test_path)

    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
    print(score)