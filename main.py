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
from datetime import datetime, timedelta
import json


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


def get_week_dates(end_date: str) -> list:
    """Get week dates on and before end_date

    Args:
        end_date (str): e.g. "2023-07-05"

    Returns:
        list: a list of dates going back for a week on and before end_date
    """
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [end_date_dt - timedelta(days=x) for x in range(7)]
    return [date.strftime("%Y-%m-%d") for date in date_list]


def read_json(json_path: str) -> list:
    """Load json and return list

    Args:
        json_path (str): path to json

    Returns:
        list: each sleep.json contains a list of dicts
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # set up vars
    dir = "./oura"
    patient = "Percept004"
    log = "./example.log"
    date = "2023-07-05"

    # set up logger
    logger = setup_logger("example_logger", "example.log")

    # locate patient folder
    patient_dir = os.path.join(dir, patient)
    if not os.path.exists(patient_dir):
        logger.error("Patient Not Found.")

    # get sleep pattern for the past week
    # going back on and before date
    dates = get_week_dates(date)
    sleeps = []
    for date in dates:
        patient_date_json = os.path.join(patient_dir, date, "sleep.json")
        if not os.path.exists(patient_date_json):
            logger.error(f"{date} sleep data not found.")
        else:
            date_list = read_json(patient_date_json)
            sleeps.extend(date_list)
    data = SleepData(sleeps)
    data.plot_sleep_distribution_for_week()
    data.plot_sleep_habit_for_week_polar()

    # print(data.plot_sleep_phase_5_min(day))
    # print(data.get_summary_stat_for_day(day))
