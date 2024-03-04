import os
import sys

from dataclasses import dataclass

from utils import evaluate_model, save_object

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score

from logger import logging
from exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting the training and test input data into X,y')
            X_train, y_train, X_test, y_test = train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]

            models = {
                'LinearRegression':LinearRegression(),
                'RandomForest':RandomForestRegressor(),
                'DecisionTree':DecisionTreeRegressor(),
                #'neighbor':KNeighborsRegressor(),
                #'GradientBoosting':GradientBoostingRegressor()
            }

            params={

                "LinearRegression":{},
                
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [128,256]
                },

                "DecisionTree": {
                    'criterion':['squared_error', 'absolute_error'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                }
                
            
            }

            logging.info('Entering Training phase')

            model_report :dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, param = params)

            logging.info('Training Complete, entering predict phase')


            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException('No best model found')
            logging.info('Best model Found')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predict = best_model.predict(X_test)
            r2_scor = r2_score(predict, y_test)
            return r2_scor

        except Exception as e:
            raise CustomException(e,sys)
        