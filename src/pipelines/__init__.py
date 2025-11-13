import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from textblob import TextBlob

from src.utils import load_obj, clean_text


class PredictPipeline:
   def __init__(self):
      pass

   def predict(self, features):
      try:
         # model path and preprocessor path
         model_path = os.path.join('artifacts', 'model.pkl')
         preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

         if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
         if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

         # load both model and preprocessor
         model = load_obj(model_path)
         preprocessor = load_obj(preprocessor_path)

         #features preprocessing
         preprocessed_features = preprocessor.transform(features)

         #predict the output
         Y_pred = model.predict(preprocessed_features)

         return Y_pred

      except Exception as e:
         raise CustomException(e, sys)
      
class CustomData:
   def __init__(self, article):
      self.text = article

   def get_data_as_dataFrame(self):
      try:
         text_len = len(self.text)
         sentiment_score = TextBlob(str(self.text)).sentiment.polarity
         cleaned_text = clean_text(self.text)

         data = {
            'clean_text': [cleaned_text],
            'text_len': [text_len],
            'sentiment':[sentiment_score]
         }

         data_frame = pd.DataFrame(data)
         return data_frame
      except Exception as e:
         raise CustomException(e, sys)
   

