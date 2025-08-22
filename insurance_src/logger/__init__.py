import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Simple path setup
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)



if __name__ == "__main__":
    logger.info("Logger initialized successfully.")
    
    
# import logging
# import os
# from logging.handlers import RotatingFileHandler

# # ======================
# #   Setup Log Directory
# # ======================
# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)

# # One main file, rotated automatically (not timestamped)
# LOG_FILE = os.path.join(LOG_DIR, "app.log")

# # ======================
# #   Setup Logger
# # ======================
# logger = logging.getLogger("insurance_logger")
# logger.setLevel(logging.INFO)

# # Avoid adding duplicate handlers if re-imported
# if not logger.handlers:
#     # Format
#     formatter = logging.Formatter(
#         "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
#     )

#     # File Handler (5 MB max, keep 3 backups)
#     file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
#     file_handler.setFormatter(formatter)

#     # Console Handler (so Airflow UI & terminal see logs too)
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)

#     # Add Handlers
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

# # Prevent log messages from being swallowed by Airflow root logger
# logger.propagate = False


# if __name__ == "__main__":
#     logger.info("Logger initialized successfully.")
