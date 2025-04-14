"""
Entry point of running notification system
Usage
- specicfy patient
- specify dir
- python main.py
"""

from sleep import Sleep
from activity import Activity
from master import Master
from utils import (
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
from trbdv0.survey_automation import send_survey, send_wearable_reminder
import argparse
import numpy as np
import json
from collections import defaultdict


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


def generate_subject_line(all_patient_stats: list) -> str:
    """
    Generate a concise subject line for doctors, listing patients needing attention,
    and briefly noting all-clear patients.

    Args:
        all_patient_stats (list): Each dict has:
            - "summary": patient summary
            - "warnings": dict of flags

    Returns:
        str: Subject line
    """
    yesterday = get_yesterdays_date()

    needs_attention = []
    all_clear = []

    for entry in all_patient_stats:
        patient = entry["summary"]["patient"]
        warnings = entry["warning"]

        if any(warnings.values()):
            needs_attention.append(patient)
        else:
            all_clear.append(patient)

    if not needs_attention:
        patients_str = ", ".join(sorted(all_clear))
        return f"[All Clear] for Patients: {patients_str} on {yesterday}"

    flagged_str = ", ".join(sorted(needs_attention))
    return f"[Warning: {flagged_str} need review] on {yesterday}"


def generate_email_body(all_patient_stats: list) -> str:
    """
    Generate an HTML table of patient summary stats with red highlights for triggered warnings.

    Args:
        all_patient_stats (list): List of dicts each containing:
            - "summary": dict from get_summary_stats()
            - "warnings": dict from generate_warning_flags()

    Returns:
        str: HTML table (as a string)
    """
    df_rows = []

    for entry in all_patient_stats:
        summary = entry["summary"]
        warnings = entry["warning"]
        patient = summary["patient"]

        def style(val, *flags):
            if pd.isna(val) or any(warnings.get(flag, False) for flag in flags):
                return f'<span style="background-color: #ff5252">{val}</span>'
            return val

        row = {
            "Patient": patient,
            "Missing Days": style(
                f"{summary['number_of_noncompliance_days']}/{summary['number_of_days']}",
                "has_noncompliance_days",
            ),
            "Average Sleep (h)": style(
                f"{summary.get('average_sleep_hours', np.nan):.1f}",
                "average_sleep_nan",
            ),
            "Yesterday's Sleep (h)": style(
                f"{summary.get('yesterday_sleep_hours', np.nan):.1f}",
                "sleep_variation",
                "yesterday_sleep_nan",
                "yesterday_sleep_less_than_5",
            ),
            "Average Steps": style(
                f"{summary.get('average_steps', np.nan):.0f}",
                "average_steps_nan",
            ),
            "Yesterday's Steps": style(
                f"{summary.get('yesterday_steps', np.nan):.0f}",
                "steps_variation",
                "yesterday_steps_nan",
            ),
            "Average MET": style(
                f"{summary.get('average_met', np.nan):.2f}",
                "average_met_nan",
            ),
            "Yesterday's MET": style(
                f"{summary.get('yesterday_met', np.nan):.2f}",
                "met_variation",
                "yesterday_met_nan",
            ),
        }

        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    html_table = df.to_html(index=False, escape=False)

    note = (
        "<p style='font-size: 0.9em; color: #555;'>"
        "<strong>Note:</strong> Cells highlighted in red indicate "
        "either <em>missing data</em> or sleep < 5 hours, or values that deviate more than ±25% from the patient's average."
        "</p>"
    )

    return html_table + note


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
    today_date = get_todays_date()

    # initialize email sender
    email_sender = EmailSender(smtp_server, smtp_port, smtp_user, smtp_password)
    email_sender.connect()

    all_patient_stats = []
    all_attachments = []

    for patient in config["active_patients"]:
        # locate patient folder
        patient_in_dir = None
        for input_dir in config["input_dir"]:
            potential_dir = os.path.join(input_dir, patient, "oura")
            if os.path.exists(potential_dir):
                patient_in_dir = potential_dir
                break

        # check if patient folder exists
        if patient_in_dir is None:
            logger.error(f"Patient folder {patient_in_dir} does not exist.")
            continue

        # set up output dir
        patient_out_dir = os.path.join(output_dir, patient, today_date)
        os.makedirs(patient_out_dir, exist_ok=True)
        log = os.path.join(patient_out_dir, "log.log")

        # set up logger
        logger = setup_logger(patient, log)

        sleep = Sleep(patient, config, patient_in_dir, patient_out_dir, logger)
        activity = Activity(patient, config, patient_in_dir, patient_out_dir, logger)
        master = Master(sleep, activity)

        patient_summary_stats = master.get_summary_stats()
        warnings = master.generate_warning_flags(patient_summary_stats)

        # send quatrics survey if sleep_variation is triggered
        if warnings["sleep_variation"]:
            logger.info(f"sleep_variation triggered, sending survey to {patient}...")
            send_survey(patient)

        # send quatrics survey if non_wear_time is triggered
        if warnings["yesterday_non_wear_time_over_8"]:
            logger.info(
                f"yesterday_non_wear_time_over_8 triggered, sending survey to {patient}..."
            )
            send_wearable_reminder(patient)

        # save summary stats to file
        summary_stats_file = os.path.join(patient_out_dir, f"{patient}.json")
        with open(summary_stats_file, "w") as f:
            json.dump(patient_summary_stats, f, indent=4)

        master.plot_combined_sleep_and_met()

        # get attachments
        attachments = get_attachments(patient_out_dir)
        all_attachments.extend(attachments)

        all_patient_stats.append(
            {
                "summary": patient_summary_stats,
                "warning": warnings,
            }
        )

        logger.info(f"Data for patient {patient} processed successfully!")

    # send a single email to recepients
    email_body = generate_email_body(all_patient_stats)
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
