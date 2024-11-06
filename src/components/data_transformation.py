import sys
import os
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from src.utils import save_object
from typing import List

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransforamtion:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformor_object(self) ->  ColumnTransformer:
        '''
        This function gives a preprocessor to transform the columns of the raw data
        '''

        try:
            numerical_columns = ["writing_score", "reading_score"]
            logging.info(f'numerical colums: {numerical_columns}')
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            logging.info(f'categorical colums: {categorical_columns}')

            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)
    
    def initiate_data_trainsformation(self, train_path: str, test_path: str) -> List[str]:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')

            logging.info('Obtaing preprocesseing object')
            preprocessing_obj = self.get_data_transformor_object()

            target_column = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Fitting and transforming the train and test data to the processor')
            input_transform_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_transform_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info('Transformation completed')

            logging.info('concatinating transformed train and test array with traget feature')
            train_arr = np.c_[
                input_transform_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_transform_test_arr, np.array(target_feature_test_df)
            ]
            logging.info("completed the concatination of features and target into train and test array")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)
        
