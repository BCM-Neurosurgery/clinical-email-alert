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
    get_missing_dates,
)
import os
import logging
import pytz
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
from trbdv0.send_email import EmailSender
import argparse


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
        sleep_data = []
        activity_data = []

        patient_date_json = os.path.join(patient_dir, date, "sleep.json")
        daily_activity_json = os.path.join(patient_dir, date, "daily_activity.json")

        if not os.path.exists(patient_date_json):
            logger.error(f"{date} sleep data not found.")
            sleep_data = []
        else:
            sleep_data = read_json(patient_date_json)

        if not os.path.exists(daily_activity_json):
            logger.error(f"{date} daily activity data not found.")
            activity_data = []
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


def get_patient_warnings(patient_stats: dict, yesterday_date: str) -> list:
    """Get all warnings for a patient's health metrics.

    Analyzes sleep and step data to generate warnings for:
    - Missing data
    - Low sleep duration (<5 hours)
    - Abnormal sleep variation (±25% from average)
    - Abnormal steps variation (±25% from average)

    Args:
        patient_stats (dict): Patient statistics containing:
            - sleep_df (pd.DataFrame): Sleep data frame
            - yesterday_sleep (float): Hours slept yesterday
            - average_sleep (float): Average sleep hours
            - yesterday_steps (float): Steps taken yesterday
            - average_steps (float): Average daily steps
        yesterday_date (str): Date string for yesterday in format 'YYYY-MM-DD'

    Returns:
        list: List of tuples (warning_message, warning_type) where warning_type is one of:
            - 'missing_data': Missing sleep data
            - 'low_sleep': Sleep duration < 5 hours
            - 'sleep_variation': Abnormal sleep pattern
            - 'steps_variation': Abnormal steps pattern
    """
    warnings = []
    sleep_df = patient_stats.get("sleep_df", pd.DataFrame())

    # Sleep data availability warnings
    if sleep_df.empty:
        warnings.append(("No Sleep Data", "missing_data"))
    elif yesterday_date not in sleep_df.index:
        warnings.append(("Missing Sleep Data", "missing_data"))

    # Sleep duration warning
    if patient_stats["yesterday_sleep"] < 5:
        warnings.append(("Sleep < 5 hours", "low_sleep"))

    # Sleep variation warning
    if (
        not pd.isna(patient_stats["yesterday_sleep"])
        and not pd.isna(patient_stats["average_sleep"])
        and patient_stats["average_sleep"] > 0
        and (
            patient_stats["yesterday_sleep"] < 0.75 * patient_stats["average_sleep"]
            or patient_stats["yesterday_sleep"] > 1.25 * patient_stats["average_sleep"]
        )
    ):
        warnings.append(("Sleep Variation", "sleep_variation"))

    # Steps variation warning
    if (
        not pd.isna(patient_stats.get("yesterday_steps"))
        and not pd.isna(patient_stats.get("average_steps"))
        and patient_stats["average_steps"] > 0
        and (
            patient_stats["yesterday_steps"] < 0.75 * patient_stats["average_steps"]
            or patient_stats["yesterday_steps"] > 1.25 * patient_stats["average_steps"]
        )
    ):
        warnings.append(("Steps Variation", "steps_variation"))

    return warnings


def generate_subject_line(all_patient_stats: list) -> str:
    """Generate an email subject line summarizing all patient warnings.

    Creates a subject line that includes warning types and affected patient names.
    If no warnings exist, shows [All Clear].

    Args:
        all_patient_stats (list): List of dictionaries containing patient stats:
            [
                {
                    "patient": str,
                    "average_sleep": float,
                    "average_steps": float,
                    "yesterday_sleep": float,
                    "yesterday_steps": float,
                    "sleep_df": pd.DataFrame
                },
                ...
            ]

    Returns:
        str: Formatted subject line in format:
            "[Warning: type1, type2] for Patients: pat1, pat2 on YYYY-MM-DD"
            or "[All Clear] for Patients: pat1, pat2 on YYYY-MM-DD"
    """
    warnings = set()
    yesterday_date = get_yesterdays_date()
    patient_list = [patient_stats["patient"] for patient_stats in all_patient_stats]

    for patient_stats in all_patient_stats:
        patient_warnings = get_patient_warnings(patient_stats, yesterday_date)
        warnings.update(warning[0] for warning in patient_warnings)

    status = (
        "[All Clear]" if not warnings else f"[Warning: {', '.join(sorted(warnings))}]"
    )
    patients_str = ", ".join(patient_list)
    return f"{status} for Patients: {patients_str} on {yesterday_date}"


def generate_email_body(missing_dates_dict, total_days, all_patients_stats) -> str:
    df_rows = []
    yesterday_date = get_yesterdays_date()

    for patient_stats in all_patients_stats:
        warnings = get_patient_warnings(patient_stats, yesterday_date)
        warning_types = [w[1] for w in warnings]

        missing_dates = missing_dates_dict.get(patient_stats["patient"], [])
        missing_count = len(missing_dates)

        row = {
            "Patient": patient_stats["patient"],
            "Missing Days": f"{missing_count}/{total_days}",
            "Average Sleep": f"{patient_stats['average_sleep']:.1f}",
            "Yesterday's Sleep": f"{patient_stats['yesterday_sleep']:.1f}",
            "Average Steps": f"{patient_stats.get('average_steps', 'N/A'):.0f}",
            "Yesterday's Steps": f"{patient_stats.get('yesterday_steps', 'N/A'):.0f}",
        }

        # Add HTML highlighting based on warning types
        if "missing_data" in warning_types:
            row["Missing Days"] = (
                f'<span style="background-color: #ff5252">{row["Missing Days"]}</span>'
            )
        if "low_sleep" in warning_types or "sleep_variation" in warning_types:
            row["Yesterday's Sleep"] = (
                f'<span style="background-color: #ff5252">{row["Yesterday\'s Sleep"]}</span>'
            )
        if "steps_variation" in warning_types:
            row["Yesterday's Steps"] = (
                f'<span style="background-color: #ff5252">{row["Yesterday\'s Steps"]}</span>'
            )

        df_rows.append(row)

    # Create DataFrame and convert to HTML
    df = pd.DataFrame(df_rows)
    return df.to_html(index=False, escape=False)


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
