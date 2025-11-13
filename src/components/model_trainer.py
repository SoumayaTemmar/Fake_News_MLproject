import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_obj

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
   model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
   def __init__(self):
      self.model_trainer_config = ModelTrainerConfig()

   def initiate_model_trainer(self,train_arr,test_arr,y_train,y_test):
      logging.info("entering model trainer module")
      try:
         #we can add other models in the future,but for now logistic reg is doing really good
         models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
         }

         # evaluate each and every model
         print("evaluating the models...")
         logging.info("evalting the models performance")
         reports = evaluate_models(train_arr, test_arr, y_train, y_test,models)

         #get the best model
         logging.info("getting the best model")

         best_obj = max(reports, key=lambda x: x['acc_score'])
         best_model_name = best_obj['name']
         model_acc_score = best_obj['acc_score']

         print(f"best model name: {best_model_name}, with acc_score: {model_acc_score}\n and cv_score: {best_obj['cv_score_mean']}")

         #get the model
         best_model = models[best_model_name]

         #save the best model
         logging.info(f"saving the best model-{best_model_name}- to {self.model_trainer_config.model_file_path}")
         save_obj(
            file_path = self.model_trainer_config.model_file_path,
            obj = best_model
         )
         
         #predict using the best model
         y_pred = best_model.predict(test_arr)
         classification_rep = classification_report(y_test, y_pred)

         return(
            classification_rep,
            self.model_trainer_config.model_file_path
         )

      except Exception as e:
         raise CustomException(e, sys)
