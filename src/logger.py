import logging
import os
from datetime import datetime

# Create a unique log file name based on the current date and time
LOG_FILE = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'

# Define the path for the logs directory within the current working directory
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)

# Ensure that the logs directory exists; if it doesn't, create it
os.makedirs(logs_path, exist_ok=True)

# Full path for the log file to be written to
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,                      # Path to log file
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO,                          # Set log level to INFO
)

if __name__ == "__main__":
    logging.info("checking logging")