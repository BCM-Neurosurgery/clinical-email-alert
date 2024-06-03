"""
Entry point of running notification system
Usage
- specicfy patient
- specify dir
- python main.py
"""

from trbdv0.sleepdata import SleepData
import os
import logging
import pytz
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import json
import pandas as pd
from trbdv0.send_email import EmailSender


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


def get_past_week_dates(end_date: str, past_days: int = 7) -> list:
    """Get week dates on and before end_date

    Args:
        end_date (str): e.g. "2023-07-05"

    Returns:
        list: a list of dates going back for a week on and before end_date
    """
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [end_date_dt - timedelta(days=x) for x in range(past_days)]
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


def get_todays_date() -> str:
    """Get date of today

    Returns:
        str: e.g. "2024-05-01"
    """
    today = datetime.today()
    return today.strftime("%Y-%m-%d")


def missing_data_on_date(sleep_data: list, date: str) -> bool:
    """Return True if missing data on date

    Args:
        sleep_data (list): [{}, {}, ...]
        date (str): "2023-07-15"

    Returns:
        bool: True if we are missing data
    """
    for data in sleep_data:
        if data["day"] == date and pd.isna(data["bedtime_start"]):
            return True
    return False


def read_config(config_file: str) -> dict:
    """Read config.json into dict

    Args:
        config_file (str): path to config file

    Returns:
        dict: loaded into dictionary
    """
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def merge_sleep_data(dates: list, patient_dir: str, logger: logging.Logger) -> list:
    """Merge sleep data based on dates, return a merged list

    Args:
        week_dates (list): e.g. ["2023-06-23", "2023-06-24", ...]
        patient_dir (str): patient dir that contains all data,
            e.g. "./oura/Percept004/"
        logger (logging.Logger): logger

    Returns:
        list: [{}, {}, {}, ..., {}]
    """
    res = []
    for date in dates:
        patient_date_json = os.path.join(patient_dir, date, "sleep.json")
        if not os.path.exists(patient_date_json):
            logger.error(f"{date} sleep data not found.")
            res.append(
                {
                    "day": date,
                    "sleep_phase_5_min": "",
                    "bedtime_start": None,
                    "bedtime_end": None,
                }
            )
        else:
            date_list = read_json(patient_date_json)
            res.extend(date_list)
    return res


def get_missing_dates(dates: list, patient_dir: str) -> list:
    """Return a list of missing dates in the dates list

    Args:
        dates (list): e.g. ["2023-06-23", "2023-06-24", ...]
        patient_dir (str): patient dir that contains all data,
            e.g. "./oura/Percept004/"

    Returns:
        list: ["2023-06-23", "2023-06-24"]
    """
    res = []
    for date in dates:
        patient_date_json = os.path.join(patient_dir, date, "sleep.json")
        if not os.path.exists(patient_date_json):
            res.append(date)
    return res


def generate_email_body(missing_dates, total_days=14):
    missing_count = len(missing_dates)
    missing_dates_str = ", ".join(missing_dates)

    email_body = (
        f"Sleep data processed successfully for the past {total_days} days.\n\n"
        f"There are {missing_count} out of {total_days} days with missing data.\n"
        f"Missing dates: {missing_dates_str}\n\n"
        f"Please see attachments for more details."
    )

    return email_body


def get_attachments(dir: str):
    """Return a list of paths to files to be attached
    to the email


    Args:
        dir (str): folder that contains the plots and log
            files
    """
    attachments = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".log"):
                attachments.append(os.path.join(root, file))
    return attachments


def main():
    # read config
    config_file = "config.json"
    config = read_config(config_file)

    # set up vars
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    email_recipients = config["email_recipients"]
    smtp_server = config["smtp_server"]
    smtp_port = config["smtp_port"]
    smtp_user = config["smtp_user"]
    smtp_password = config["smtp_password"]
    past_days = config["past_days"]
    # date = get_todays_date()
    date = get_todays_date()

    # initialize email sender
    email_sender = EmailSender(smtp_server, smtp_port, smtp_user, smtp_password)
    email_sender.connect()

    for patient in config["active_patients"]:
        # locate patient folder
        patient_in_dir = os.path.join(input_dir, patient)

        # check if patient folder exists
        if not os.path.exists(patient_in_dir):
            logger.error(f"Patient folder {patient_in_dir} does not exist.")
            continue

        # set up output dir
        patient_out_dir = os.path.join(output_dir, patient, date)
        os.makedirs(patient_out_dir, exist_ok=True)
        log = os.path.join(patient_out_dir, "log.log")

        # set up logger
        logger = setup_logger(patient, log)

        # get sleep pattern for the past week
        # going back on and before date
        past_week_dates = get_past_week_dates(date, past_days)
        sleeps = merge_sleep_data(past_week_dates, patient_in_dir, logger)
        data = SleepData(sleeps)

        # get summary of today's data
        if missing_data_on_date(sleeps, date):
            logger.error(f"Missing data for today {date}")
        else:
            data.get_summary_plot_for_date(date, patient_out_dir)

        # get summary for past week
        data.plot_sleep_distribution(patient_out_dir, past_days)
        data.plot_sleep_habit_polar(patient_out_dir, past_days)
        logger.info(
            f"sleep distribution plot for past {past_days} days saved to {patient_out_dir}."
        )
        logger.info(
            f"sleep habit plot for past {past_days} days saved to {patient_out_dir}."
        )

        # get attachments
        attachments = get_attachments(patient_out_dir)

        # get non-compliant dates for the past week
        missing_dates = get_missing_dates(past_week_dates, patient_in_dir)
        # send a single email to recepients
        email_body = generate_email_body(missing_dates, past_days)
        subject = f"Sleep Data Processing Successful for Patient {patient}"
        email_sender.send_email(email_recipients, subject, email_body, attachments)
        logger.info(f"Email for patient {patient} sent successfully!")

    email_sender.disconnect()


if __name__ == "__main__":
    main()
