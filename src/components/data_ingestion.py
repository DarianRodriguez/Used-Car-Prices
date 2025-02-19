import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import FeatureTransformation
from src.components.data_pipeline import PipelineBuilder
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):

        try:
            logging.info('Reading the dataset file')
            df = pd.read_csv('./artifacts/data.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            # Split the dataset (90% training + valid, 10% testing)
            train_set,test_set=train_test_split(df,test_size=0.1) #random_state=32

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    print("Data Ingestion...")
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()

    train_data = pd.read_csv(train_data_path)

    print("Feature Engineering...")
    feature_transformer = FeatureTransformation()
    train_modified = feature_transformer.transform_data(train_data)

    print(train_modified.head())

    print("\nSplitting Data...")
    X_train = train_modified.drop(columns=['price'])
    y_train = train_modified['price']

    print("Model Training...")
    pipeline_builder = PipelineBuilder()
    prep_pipeline = pipeline_builder.get_full_pipeline() #preprocessor pipeline

    trainer = ModelTrainer(prep_pipeline,n_splits=5)
    best_model, best_model_name = trainer.select_best_model(X_train, y_train, n_trials=15)




