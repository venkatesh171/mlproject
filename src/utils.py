import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import pickle
from src.logger import logging
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from typing import Dict


def save_object(file_path: str, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            logging.info("created the pickle file")

    except Exception as e:
        logging.error(CustomException(e, sys))
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(CustomException(e, sys))
        raise CustomException(e, sys)


def evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, BaseEstimator],
    params: dict,
) -> Dict[str, float]:
    try:
        report = {}

        for key, value in models.items():
            model = value
            param = params[key]

            logging.info(f"finding params for model: {key}")
            gs = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1, cv=3)
            gs.fit(X_train, y_train)

            logging.info("fitting the model with train data")
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            logging.info("model training completed")

            # y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            logging.info(f"model: {key} r2_score is: {test_model_score}")
            report[key] = test_model_score
        logging.info("models evaluation completed")
        return report

    except Exception as e:
        logging.error(CustomException(e, sys))
        raise CustomException(e, sys)
