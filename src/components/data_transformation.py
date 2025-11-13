import sys
import os
import pandas as pd
import numpy as np
from src.utils import save_obj, clean_text
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from src.components.model_trainer import ModelTrainer


@dataclass
class DataTransformationConfig:
   preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
   def __init__(self):
      self.data_transformation_config = DataTransformationConfig()

      
   def get_transformer_obj(self):
      try:
         
         text_column = 'clean_text'
         numeric_columns = ['text_len','sentiment']

         text_pipeline = Pipeline(
            steps=[
               ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)))
            ]
         )

         numeric_pipeline = Pipeline(
            steps=[
               ('scaler', StandardScaler(with_mean=False))
            ]
         )

         preprocessor = ColumnTransformer(
            [
               ('text_pipeline', text_pipeline, text_column),
               ('numeric_pipeline', numeric_pipeline, numeric_columns)
            ]
         )

         return preprocessor

      except Exception as e:
         raise CustomException(e, sys)
      
   def initiate_data_transformation(self, train_path:str, test_path:str):
      logging.info("entering the data transformation module")
      try:
         
         #read train and test data
         train_df = pd.read_csv(train_path)
         test_df = pd.read_csv(test_path)

         #clean the text feature
         print("cleaning text feature...")
         train_df['clean_text'] = train_df['text'].apply(clean_text)
         test_df['clean_text'] = test_df['text'].apply(clean_text)

         #seprate the datasets into inputs and label
         input_features_train_df = train_df.drop(columns=['label'],axis=1)
         target_feature_train_df = train_df['label']

         input_feature_test_df = test_df.drop(columns=['label'],axis=1)
         target_feature_test_df = test_df['label']

         #get the preprocessor obj
         preprocessor = self.get_transformer_obj()

         #transform the data
         print("transforming data...")
         input_features_train_arr = preprocessor.fit_transform(input_features_train_df)
         input_features_test_arr = preprocessor.transform(input_feature_test_df)


         #save the preprocessor obj
         save_obj(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor
         )

         return(
            input_features_train_arr,
            input_features_test_arr,
            target_feature_train_df,
            target_feature_test_df,
            self.data_transformation_config.preprocessor_obj_file_path
         )
      except Exception as e:
         raise CustomException(e, sys)
      
if __name__ == "__main__":
   data_transformation = DataTransformation()
   trainPath = os.path.join('artifacts', 'train.csv')
   testPath = os.path.join('artifacts', 'test.csv')

   train_arr, test_arr,y_train,y_test,_ = data_transformation.initiate_data_transformation(trainPath, testPath)

   #training the model
   model_trainer = ModelTrainer()
   classification_report, _ = model_trainer.initiate_model_trainer(train_arr, test_arr, y_train, y_test)
   print(f"\n classification report: \n {classification_report}")


