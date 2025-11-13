import sys
import os
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
   def __init__(self):
      self.data_ingestion = DataIngestion()
      self.data_transformation = DataTransformation()
      self.model_trainer = ModelTrainer()

   def init_train_pipeline(self):
      logging.info("starting training pipeline")
      try:
         logging.info("starting data ingestion")
         dataset_path = os.path.join('Fake_News_noteBook', 'data', 'WELFake_Dataset.csv')
         train_path, test_path = self.data_ingestion.initiate_data_ingestion(dataset_path)
         logging.info("data ingestion completed")

         logging.info("starting data transformation")
         train_arr, test_arr, y_train, y_test,_ = self.data_transformation.initiate_data_transformation(train_path, test_path)
         logging.info("data transformation completed")

         logging.info("starting model training")
         classification_report, _ = self.model_trainer.initiate_model_trainer(train_arr, test_arr, y_train, y_test)
         print(f"\n classification report: \n {classification_report}")

         logging.info("model training completed")

      except Exception as e:
         raise CustomException(e, sys)
      
if __name__ == "__main__":
   train_pipeline = TrainPipeline()
   train_pipeline.init_train_pipeline()
