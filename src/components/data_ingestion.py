import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_path:str=os.path.join('Artifacts','train.csv')
    test_path:str=os.path.join('Artifacts','test.csv')
    raw_data_path:str=os.path.join('Artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("enter data ingestion method")
        try:
            df=pd.read_csv('notebook\stud.csv')
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion.train_path),exist_ok=True)
            df.to_csv(self.data_ingestion.raw_data_path,index=False,header=True)

            logging.info("train_test_split initiate")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.data_ingestion.train_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion.test_path,index=False,header=True)
            logging.info('data ingestion completed')
            
            return(self.data_ingestion.train_path,
                   self.data_ingestion.test_path
                   )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initite_data_transformation(train_data,test_data)

    model_trainer =  ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
