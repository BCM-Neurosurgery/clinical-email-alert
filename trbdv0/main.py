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
import argparse
import numpy as np


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


def get_past_dates(end_date: str, past_days: int = 7) -> list:
    """Get week dates on and before end_date

    Args:
        end_date (str): e.g. "2023-07-05"

    Returns:
        list: a list of dates going back for a week on and before end_date
    """
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [end_date_dt - timedelta(days=x) for x in range(1, past_days + 1)]
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
    """Merge sleep data based on dates, return a merged list with selected keys

    Args:
        dates (list): e.g. ["2023-06-23", "2023-06-24", ...]
        patient_dir (str): patient dir that contains all data,
            e.g. "./oura/Percept004/"
        logger (logging.Logger): logger

    Returns:
        list: [{}, {}, {}, ..., {}]
    """
    res = []
    for date in dates:
        patient_date_json = os.path.join(patient_dir, date, "sleep.json")
        daily_activity_json = os.path.join(patient_dir, date, "daily_activity.json")

        if not os.path.exists(patient_date_json):
            logger.error(f"{date} sleep data not found.")
            res.append(
                {
                    "day": date,
                    "sleep_phase_5_min": "",
                    "bedtime_start": None,
                    "bedtime_end": None,
                    "class_5_min": "",
                    "non_wear_time": 0,
                    "timestamp": "",
                }
            )
        else:
            sleep_data = read_json(patient_date_json)
            activity_data = (
                read_json(daily_activity_json)
                if os.path.exists(daily_activity_json)
                else []
            )

            for sleep_entry in sleep_data:
                entry = {
                    "day": date,
                    "sleep_phase_5_min": sleep_entry.get("sleep_phase_5_min", ""),
                    "bedtime_start": sleep_entry.get("bedtime_start", None),
                    "bedtime_end": sleep_entry.get("bedtime_end", None),
                    "class_5_min": "",
                    "non_wear_time": 0,
                    "timestamp": "",
                }

                # Find corresponding activity entry with the same day
                matching_activity = next(
                    (
                        activity
                        for activity in activity_data
                        if activity.get("day") == date
                    ),
                    {},
                )
                entry["class_5_min"] = matching_activity.get("class_5_min", "")
                entry["non_wear_time"] = matching_activity.get("non_wear_time", 0)
                entry["timestamp"] = matching_activity.get("timestamp", "")

                res.append(entry)
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


def generate_subject_line(patient: str, stats: dict, today_date: str) -> str:
    """Generate email subject line

    Args:
        patient (str): Patient identifier
        stats (dict): {"average_sleep": float, "sleep_df":
            DataFrame with dates as index and "Total Sleep" for each date}
        today_date (str): Today's date in the format "2024-06-01"

    Returns:
        str: Subject line of the email
    """
    sleep_df = stats.get("sleep_df", pd.DataFrame())
    yesterday_date = (
        datetime.strptime(today_date, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if sleep_df.empty:
        status = "[Warning: No Sleep Data]"
    elif yesterday_date not in sleep_df.index:
        status = "[Warning: Missing Sleep Data]"
    elif sleep_df.loc[yesterday_date, "Total Sleep"] < 5:
        status = "[Warning: Sleep < 5 hours]"
    else:
        status = "[All Clear]"

    subject = f"{status} for Patient {patient} on {yesterday_date}"
    return subject


def generate_email_body(missing_dates, total_days, stats) -> str:
    """Generate email body

    Args:
        missing_dates (list): ["2024-06-01", ...]
        total_days (int): number of total days
        stats (dict): {"average_sleep": float, "sleep_df":
            DataFrame with dates as index and "Total Sleep" for each date}

    Returns:
        str: a string of email body
    """
    missing_count = len(missing_dates)

    sleep_df = stats.get("sleep_df", pd.DataFrame())
    if sleep_df.empty:
        low_sleep_days = np.nan
    else:
        low_sleep_days = (sleep_df["Total Sleep"] < 5).sum()

    df = pd.DataFrame(
        {
            "Average Sleep (hours)": [stats["average_sleep"]],
            "Non-Compliance Days": [missing_count],
            "Days with < 5 hours Sleep": [low_sleep_days],
        }
    )

    # Convert DataFrame to HTML table with borders
    df_html = df.to_html(index=False, border=1, justify="center")

    # Create missing dates section
    missing_dates_html = "<br>".join(missing_dates)
    missing_dates_section = f"""
    <p>Missing Dates:</p>
    <p>{missing_dates_html}</p>
    """

    email_body = f"""
    <html>
        <body>
            <p>Sleep data processed successfully for the past {total_days} days.</p>
            {df_html}
            {missing_dates_section}
            <p>Please see attachments for more details.</p>
        </body>
    </html>
    """

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


def main(config_file):
    config = read_config(config_file)

    # set up vars
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    email_recipients = config["email_recipients"]
    smtp_server = config["smtp_server"]
    smtp_port = config["smtp_port"]
    smtp_user = config["smtp_user"]
    smtp_password = config["smtp_password"]
    num_past_days = config["past_days"]
    today_date = get_todays_date()

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
        patient_out_dir = os.path.join(output_dir, patient, today_date)
        os.makedirs(patient_out_dir, exist_ok=True)
        log = os.path.join(patient_out_dir, "log.log")

        # set up logger
        logger = setup_logger(patient, log)

        past_dates = get_past_dates(today_date, num_past_days)
        sleeps = merge_sleep_data(past_dates, patient_in_dir, logger)
        data = SleepData(sleeps)

        # get summary for past x days
        stats = data.plot_combined_sleep_plots(patient_out_dir)
        logger.info(
            f"sleep combined plot for past {num_past_days} days saved to {patient_out_dir}."
        )

        # get attachments
        attachments = get_attachments(patient_out_dir)

        # get non-compliant dates for the past x days
        missing_dates = get_missing_dates(past_dates, patient_in_dir)
        # send a single email to recepients
        email_body = generate_email_body(missing_dates, num_past_days, stats)
        subject = generate_subject_line(patient, stats, today_date)
        email_sender.send_email(email_recipients, subject, email_body, attachments)
        logger.info(f"Email for patient {patient} sent successfully!")

    email_sender.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the notification system.")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to the config file"
    )
    args = parser.parse_args()

    main(args.config)
