# below code is to check the logging config
import sys

from src.exception import MyException
# below code is to check the exception config
from src.logger import logging
from src.pipline.training_pipeline import TrainPipeline

# logging.debug("This is a debug message.")
# logging.info("This is an info message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")
# logging.critical("This is a critical message.")


#---------------------------------------------------------------------


# try:
#     a = 1+'Z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e



pipeline = TrainPipeline()
pipeline.run_pipeline()