import pandas as pd
import numpy as np
from dataclasses import dataclass
import os 
import sys
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(
            self, train_array: np.ndarray, 
            test_array: np.ndarray) -> float:
        try:
            logging.info('Split train and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                'Decision Tress': DecisionTreeRegressor(),
                'Linear Regission': LinearRegression(),
                'K-Neighbour Classifier': KNeighborsRegressor(),
                'XGB Regressor': XGBRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # logging.info('training the best model')
            best_model = models[best_model_name]
            # best_model.fit(X_train, y_train)
            # logging.info('training completed')

            if best_model_score < 0.6:
                logging.info('No Best model found!')
                raise CustomException("No Best model found!")
            
            logging.info('pickling the best model')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            logging.info(f'pickle file created at {self.model_trainer_config.trained_model_file_path}')

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)