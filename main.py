"""
Entry point of running notification system
Usage
- specicfy patient
- specify dir
- python main.py
"""

from sleepdata import SleepData
import os
import logging
import pytz
from logging.handlers import RotatingFileHandler
from datetime import datetime


# Custom formatter class to handle timezone
class CSTFormatter(logging.Formatter):
    """Logging Formatter to add timestamps in the US Central timezone."""

    def converter(self, timestamp):
        cst_time = pytz.timezone("America/Chicago")
        return datetime.fromtimestamp(timestamp, cst_time)

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat()


# Function to setup logger
def setup_logger(name, file_path, level=logging.INFO):
    """Setup logger with custom timezone formatter."""
    formatter = CSTFormatter("%(asctime)s - %(levelname)s - %(message)s")
    # File handler
    file_handler = RotatingFileHandler(file_path, maxBytes=1048576, backupCount=5)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    # set up vars
    dir = "./oura"
    patient = "Percept005"
    log = "./example.log"

    # set up logger
    logger = setup_logger("example_logger", "example.log")
    logger.info("This is an info message")
    logger.error("This is an error message")

    # locate patient folder
    patient_dir = os.path.join(dir, patient)
    if not os.path.exists(patient_dir):
        logger.error("Patient Not Found.")
        logger.info("what")

    # path = "Percept009_Sleep.json"
    # data = SleepData(path)
    # day = "2023-09-14"
    # # print(data.get_available_days())
    # # print(data.plot_sleep_interval_on_day("2023-09-14"))
    # print(data.plot_sleep_phase_5_min(day))
    # print(data.get_summary_stat_for_day(day))
