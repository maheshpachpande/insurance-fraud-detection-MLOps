import sys
import traceback
from typing import Optional

class ErrorDetailFormatter:
    """
    Formats detailed error messages from exceptions and traceback.
    SRP: This class only handles formatting of error messages.
    """
    @staticmethod
    def format_error_message(error: Exception, exc_info: Optional[tuple] = None) -> str:
        if exc_info is None:
            exc_info = sys.exc_info()

        _, _, exc_tb = exc_info
        if exc_tb is None:
            return str(error)

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error in script: [{file_name}] line [{line_number}] message [{error}]"

class CustomException(Exception):
    """
    Custom exception class that integrates with ErrorDetailFormatter.
    SRP: Holds and returns an error message for exceptions.
    OCP: If the error formatting changes, we update ErrorDetailFormatter, not this class.
    """
    def __init__(self, error, exc_info: Optional[tuple] = None):
        # Handle both string messages and existing Exception objects
        if isinstance(error, str):
            error_to_format = Exception(error)
        elif isinstance(error, Exception):
            error_to_format = error
        else:
            error_to_format = Exception("An unexpected error type was provided.")
            
        self.error_message = ErrorDetailFormatter.format_error_message(error_to_format, exc_info)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
    
    
    
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        raise CustomException(e)