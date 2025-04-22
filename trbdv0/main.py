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
    read_config,
    validate_config_keys,
)
import os
import logging
import pytz
from logger_setup import setup_logger
from datetime import datetime
from send_email import (
    EmailSender,
    generate_subject_line,
    generate_email_body,
    get_attachments,
)
from survey_automation import send_survey, send_wearable_reminder
import argparse
import json


def main(config_file):
    config = read_config(config_file)

    # verify keys
    try:
        validate_config_keys(config)
    except KeyError as e:
        print(f"[ERROR] Config validation failed: {e}")
        return

    # set up vars
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    log_dir = config["log_dir"]
    email_recipients = config["email_recipients"]
    smtp_server = config["smtp_server"]
    smtp_port = config["smtp_port"]
    smtp_user = config["smtp_user"]
    smtp_password = config["smtp_password"]
    timezone = config["timezone"]
    quatrics_patients = config["quatrics_patients"]
    timestamp = datetime.now(pytz.timezone(timezone))
    timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    today_date = get_todays_date()

    # initialize email sender
    email_sender = EmailSender(smtp_server, smtp_port, smtp_user, smtp_password)
    email_sender.connect()

    # initialize logger
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(
        "oura-email-notification",
        os.path.join(log_dir, f"{timestamp}.log"),
        tz=timezone,
        level=logging.INFO,
    )

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

        sleep = Sleep(patient, config, patient_in_dir, patient_out_dir, logger)
        activity = Activity(patient, config, patient_in_dir, patient_out_dir, logger)
        master = Master(sleep, activity)

        patient_summary_stats = master.get_summary_stats()
        warnings = master.generate_warning_flags(patient_summary_stats)

        # send quatrics survey if sleep_variation is triggered
        if warnings["sleep_variation"] or warnings["yesterday_sleep_less_than_6"]:
            if patient in quatrics_patients:
                logger.info(
                    f"sleep_variation or yesterday_sleep_less_than_6 triggered, sending survey to {patient}..."
                )
                send_survey(patient)

        # send quatrics survey if non_wear_time is triggered
        if warnings["yesterday_non_wear_time_over_8"]:
            if patient in quatrics_patients:
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
