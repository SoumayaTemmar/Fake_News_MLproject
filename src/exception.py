import sys

def get_error_details(error, erro_details:sys):

   _,_,exc_tb = erro_details.exc_info()
   file_name = exc_tb.tb_frame.f_code.co_filename
   error_message = f"error occured in script: {file_name} at line number: {exc_tb.tb_lineno} error: {str(error)}"

   return error_message


class CustomException(Exception):
   def __init__(self, error, erro_details:sys):
      super().__init__(error)
      self.error_msg = get_error_details(error, erro_details)

   def __str__(self):
      return self.error_msg
