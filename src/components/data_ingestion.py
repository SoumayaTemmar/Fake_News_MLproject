from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from langdetect import detect, DetectorFactory
import sys
import os

from dataclasses import dataclass

DetectorFactory.seed = 0

@dataclass
class DataIngestionConfig:
   data_train_path = os.path.join('artifacts', 'train.csv')
   data_test_path = os.path.join('artifacts', 'test.csv')
   raw_data_path = os.path.join('artifacts', 'data.csv')
   clean_data_path = os.path.join('artifacts', 'clean_data.csv')

class DataIngestion:
   def __init__(self):
      self.data_ingestion_config = DataIngestionConfig()

   #detect the language of a given text
   def detect_language(self, text):
      try:
         text = str(text).strip()
         if not text:  # empty string
            return 'unknown'
         return detect(text)
      
      except Exception:
         return 'unknown'
      
   #compute the sentiment score for a given text
   def compute_sentiment_score(self, text):
      try:
         return TextBlob(str(text)).sentiment.polarity
      except Exception as e:
         raise CustomException(e, sys)
      
   #performs cleaning, filtering and feature creation
   def clean_data(self, df:pd.DataFrame)-> pd.DataFrame:
      try:
         #remove duplicates
         print('removing duplicates')
         df = df.drop_duplicates(subset=['text'])
         #remove missing values
         print('removing null values')
         df = df.dropna(subset=['text']).reset_index(drop=True)

         # add helpful diagnostics and sentiment score
         print('computing helpful diagnostics')
         df['text_len'] = df['text'].apply(len)
         df['word_count'] = df['text'].apply(lambda x: len(x.split()))

         print('calculating the sentiment score')
         df['sentiment'] = df['text'].apply(self.compute_sentiment_score)

         #detecting the language
         print('detecting the language')
         df['lang'] = df['text'].apply(self.detect_language)

         #keep only english
         print('remove non -en-')
         df = df[df['lang']=='en']

         return df
      except Exception as e:
         raise CustomException(e, sys)
         

   def initiate_data_ingestion(self, data_path:str):
      logging.info("entring data ingestion module")
      try:
         #get the data
         raw_data = pd.read_csv(data_path)
         os.makedirs(os.path.dirname(self.data_ingestion_config.clean_data_path),exist_ok=True)

         raw_data.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
         logging.info(f"saved raw data to : {self.data_ingestion_config.raw_data_path}")

         #cleaning the data
         df_clean = self.clean_data(raw_data)
         df_clean.to_csv(self.data_ingestion_config.clean_data_path,index=False,header=True)
         logging.info(f"cleaned data saved to : {self.data_ingestion_config.clean_data_path}")

         #split the data
         train_set, test_set = train_test_split(df_clean, test_size=0.2, random_state=42,stratify=df_clean['label'])

         train_set.to_csv(self.data_ingestion_config.data_train_path, index=False, header=True)
         test_set.to_csv(self.data_ingestion_config.data_test_path, index=False, header=True)

         logging.info(f"Train data saved to {self.data_ingestion_config.data_train_path}")
         logging.info(f"Test data saved to {self.data_ingestion_config.data_test_path}")

         return(
            self.data_ingestion_config.data_train_path,
            self.data_ingestion_config.data_test_path
         )
      except Exception as e:
         raise CustomException(e,sys)


if __name__ == '__main__':
   data_Ingetion = DataIngestion()
   train_path, test_path = data_Ingetion.initiate_data_ingestion('Fake_News_NoteBook\data\WELFake_Dataset.csv')

