import sys
import logging

# Function to extract detailed error information
def error_message_detail(error: Exception, error_detail: sys) -> str:
    # Extract the traceback object from the system exception info
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the filename where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Format a detailed error message with the filename, line number, and error message
    error_message = f"Error occurred in python script: {file_name} line number [{exc_tb.tb_lineno}] error message {str(error)}"
    return error_message

# Custom exception class inheriting from the base Exception class
class Custom_exception(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        # Initialize the base Exception class with the error message
        super().__init__(error_message)
        
        # Generate a detailed error message using the provided function
        self.error_message = error_message_detail(error=error_message, error_detail=error_detail)

    # Define string representation to return the detailed error message
    def __str__(self):
        return self.error_message
