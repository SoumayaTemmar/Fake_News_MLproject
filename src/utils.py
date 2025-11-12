import os
import sys
from src.exception import CustomException
import dill

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources if they don't exist
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def save_obj(file_path, obj):
   try:
      dir_path = os.path.dirname(file_path)
      os.makedirs(dir_path,exist_ok=True)

      with open(file_path, 'wb') as file_obj:
         dill.dump(obj, file_obj)
   except Exception as e:
      raise CustomException(e, sys)

def clean_text( text):
   try:
      # Lowercase
      text = str(text).lower()
      # Remove URLs, punctuation, numbers
      text = re.sub(r"http\S+|www\S+|https\S+", "", text)
      text = re.sub(r"[^a-z\s]", "", text)
      # Tokenize and remove stopwords
      tokens = [word for word in text.split() if word not in stop_words]
      # Lemmatize
      tokens = [lemmatizer.lemmatize(word) for word in tokens]
      return " ".join(tokens)
   except Exception as e:
      raise CustomException(e, sys)
   
