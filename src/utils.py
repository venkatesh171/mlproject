import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import pickle
from src.logger import logging



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            logging.info('created the pickle file')

    except Exception as e:
        logging.error(CustomException(e, sys))
        raise CustomException(e, sys)