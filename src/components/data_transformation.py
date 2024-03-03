import numpy as np 
import pandas as pd
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from logger import logging
from utils import save_object
from exception import CustomException

from dataclasses import dataclass

@dataclass
class DataTransformationConfig():
    preprocessor_file_obj = os.path.join('artifact','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numeric_columns = [
                'carat',
                'depth',
                'table',
                'x',
                'y',
                'z'
            ]
            categorical_columns = [
                'cut',
                'color',
                'clarity'
            ]

            numerical_pipeline = Pipeline(steps = [
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder())
            ]) 

            logging.info('Numeric columns transformed')
            logging.info('Categorical columns transformed')

            preprocessor = ColumnTransformer([
                ('numeric', numerical_pipeline, numeric_columns),
                ('categorical', categorical_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_object = self.get_data_transformer_obj()

            target_columns = 'price'

            train_input_feature_df = train_df.drop(columns = [target_columns], axis = 1)
            train_target_feature_df = train_df[target_columns]
            
            test_input_feature_df = test_df.drop(columns = [target_columns], axis = 1)
            test_target_feature_df = test_df[target_columns]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_train_arr = preprocessor_object.fit_transform(train_input_feature_df)
            input_test_arr = preprocessor_object.transform(test_input_feature_df)

            train_arr = np.c_[input_train_arr, np.array(train_target_feature_df)]
            test_arr = np.c_[input_test_arr, np.array(test_target_feature_df)]

            logging.info('Saving preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_file_obj,
                obj = preprocessor_object
            )

            logging.info('transformation completed')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_obj
            )

        except Exception as e:
            raise CustomException(e,sys)
        