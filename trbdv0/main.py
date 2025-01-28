"""
Entry point of running notification system
Usage
- specicfy patient
- specify dir
- python main.py
"""

from trbdv0.sleepdata import SleepData
from trbdv0.utils import (
    get_todays_date,
    get_yesterdays_date,
    get_past_dates,
    read_json,
    read_config,
)
import os
import logging
import pytz
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
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


def merge_sleep_data(dates: list, patient_dir: str, logger: logging.Logger) -> list:
    """Merge sleep data based on dates, return a merged list with selected keys.
    If sleep data is missing, fill in with empty values.

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

        if not os.path.exists(daily_activity_json):
            logger.error(f"{date} daily activity data not found.")
            res.append(
                {
                    "day": date,
                    "class_5_min": "",
                    "non_wear_time": 0,
                    "timestamp": "",
                    "steps": 0,
                }
            )
        else:
            activity_data = read_json(daily_activity_json)

        for sleep_entry in sleep_data:
            entry = {
                "day": date,
                "sleep_phase_5_min": sleep_entry.get("sleep_phase_5_min", ""),
                "bedtime_start": sleep_entry.get("bedtime_start", None),
                "bedtime_end": sleep_entry.get("bedtime_end", None),
                "class_5_min": "",
                "non_wear_time": 0,
                "timestamp": "",
                "steps": 0,
            }

            # Find corresponding activity entry with the same day
            matching_activity = next(
                (activity for activity in activity_data if activity.get("day") == date),
                {},
            )
            entry.update(
                {
                    key: matching_activity.get(key, entry[key])
                    for key in ["class_5_min", "non_wear_time", "timestamp", "steps"]
                }
            )

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


def generate_subject_line(all_patient_stats: list) -> str:
    """Generate a single status for all patients in the email subject line and list patient names

    Args:
        all_patient_stats (list): [{"patient": "Percept010", "average_sleep": float, "average_steps": float}, ...]

    Returns:
        str: A single status subject line for the email listing all patients
    """
    overall_status = "[All Clear]"
    yesterday_date = get_yesterdays_date()
    patient_list = [patient_stats["patient"] for patient_stats in all_patient_stats]

    # Iterate over all patients and update overall_status if any warning occurs
    for patient_stats in all_patient_stats:
        sleep_df = patient_stats.get("sleep_df", pd.DataFrame())

        if sleep_df.empty:
            overall_status = "[Warning: No Sleep Data]"
            break
        elif yesterday_date not in sleep_df.index:
            overall_status = "[Warning: Missing Sleep Data]"
        elif patient_stats["yesterday_sleep"] < 5:
            overall_status = "[Warning: Sleep < 5 hours]"
        elif (
            not pd.isna(patient_stats["yesterday_sleep"])
            and not pd.isna(patient_stats["average_sleep"])
            and patient_stats["average_sleep"] > 0
            and (
                patient_stats["yesterday_sleep"] < 0.75 * patient_stats["average_sleep"]
                or patient_stats["yesterday_sleep"]
                > 1.25 * patient_stats["average_sleep"]
            )
        ):
            overall_status = "[Warning: Sleep Variation]"
        elif (
            not pd.isna(patient_stats.get("yesterday_steps"))
            and not pd.isna(patient_stats.get("average_steps"))
            and patient_stats["average_steps"] > 0
            and (
                patient_stats["yesterday_steps"] < 0.75 * patient_stats["average_steps"]
                or patient_stats["yesterday_steps"]
                > 1.25 * patient_stats["average_steps"]
            )
        ):
            overall_status = "[Warning: Steps Variation]"

        # If any warning is found, stop the loop as it is the most critical
        if overall_status != "[All Clear]":
            break

    # Convert patient list to a comma-separated string
    patients_str = ", ".join(patient_list)

    # Return the overall subject line including the patient names
    subject = f"{overall_status} for Patients: {patients_str} on {yesterday_date}"
    return subject


def generate_email_body(missing_dates_dict, total_days, all_patients_stats) -> str:
    """Generate email body with an additional column for 'Yesterday's Sleep'

    Args:
        missing_dates_dict (dict): {"patient1": ["2024-06-01", ...], "patient2": [...]}
        total_days (int): number of total days
        all_patients_stats (list): [{"patient": "patient1", "average_sleep": float, "average_steps": float}, ...]

    Returns:
        str: a string of email body
    """
    df_rows = []

    for patient_stats in all_patients_stats:
        missing_dates = missing_dates_dict.get(patient_stats["patient"], [])
        missing_count = len(missing_dates)

        sleep_df = patient_stats.get("sleep_df", pd.DataFrame())

        if sleep_df.empty:
            low_sleep_days = np.nan
            yesterdays_sleep = np.nan
            yesterday_step = np.nan
        else:
            low_sleep_days = (sleep_df["Total Sleep"] < 5).sum()
            yesterdays_sleep = patient_stats.get("yesterday_sleep")
            yesterday_step = patient_stats.get("yesterday_steps")

        # Generate row for each patient
        row = {
            "Patient": patient_stats["patient"],
            "Average Sleep (hours)": patient_stats.get("average_sleep", np.nan),
            "Average Steps": patient_stats.get("average_steps", np.nan),
            "Non-Compliance Days": missing_count,
            "Days with < 5 hours Sleep": low_sleep_days,
            "Yesterday's Sleep (hours)": yesterdays_sleep,
            "Yesterday's Step Count": yesterday_step,
        }

        df_rows.append(row)

    # Create DataFrame with all patient rows
    df = pd.DataFrame(df_rows)

    # Convert DataFrame to HTML table with conditional formatting
    df_html = (
        df.style.format(
            {
                "Average Sleep (hours)": "{:.2f}",
                "Average Steps": "{:.0f}",
                "Yesterday's Sleep (hours)": "{:.2f}",
            }
        )
        .map(
            lambda val: (
                "background-color: red"
                if pd.isna(val) or (isinstance(val, (int, float)) and val < 5)
                else ""
            ),
            subset=["Yesterday's Sleep (hours)"],
        )
        .set_table_styles(
            [
                {
                    "selector": "th, td",
                    "props": [("border", "1px solid black"), ("padding", "8px")],
                },
                {
                    "selector": "table",
                    "props": [
                        ("border-collapse", "collapse"),
                        ("width", "100%"),
                        ("border", "1px solid black"),
                    ],
                },
            ]
        )
        .to_html(index=False, border=0, justify="center", escape=False)
    )

    df_html = df_html.replace(
        "<table ",
        '<table style="border-collapse: collapse; width: 100%; border: 1px solid black;" ',
    )

    # Create missing dates section for each patient
    missing_dates_section = ""
    for patient, missing_dates in missing_dates_dict.items():
        if missing_dates:
            missing_dates_html = "<br>".join(missing_dates)
            missing_dates_section += f"""
            <p><b>Patient {patient}</b> - Missing Dates:</p>
            <p>{missing_dates_html}</p>
            """

    # Generate the final email body
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
        dir (str): folder that contains the plots files
    """
    attachments = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".png"):
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

    missing_dates_dict = {}
    all_patient_stats = []
    all_attachments = []

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
        data = SleepData(patient, sleeps)

        # get summary for past x days
        stats = data.summary_stats
        data.plot_combined_sleep_plots(patient_out_dir)
        logger.info(
            f"sleep combined plot for past {num_past_days} days saved to {patient_out_dir}."
        )

        # get attachments
        attachments = get_attachments(patient_out_dir)
        all_attachments.extend(attachments)

        # get non-compliant dates for the past x days
        missing_dates = get_missing_dates(past_dates, patient_in_dir)

        missing_dates_dict[patient] = missing_dates
        all_patient_stats.append(stats)

        logger.info(f"Data for patient {patient} processed successfully!")

    # send a single email to recepients
    email_body = generate_email_body(
        missing_dates_dict, num_past_days, all_patient_stats
    )
    subject = generate_subject_line(all_patient_stats)
    email_sender.send_email(email_recipients, subject, email_body, all_attachments)
    logger.info(f"Email for patient(s) {config['active_patients']} sent successfully!")
    email_sender.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the notification system.")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to the config file"
    )
    args = parser.parse_args()

    main(args.config)
