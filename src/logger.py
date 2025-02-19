import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) #/current/working/directory/logs/11_29_2024_10_45_30.log
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH= os.path.join(logs_path,LOG_FILE)

# Configuring the Logging Modulem, logs only log messages of severity INFO or higher
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,


)