"""
Entry point of running notification system
Usage
- specicfy patient
- specify dir
- python main.py
"""

from trbdv0.sleep import Sleep
from trbdv0.activity import Activity
from trbdv0.master import Master
from trbdv0.utils import (
    get_todays_date,
    read_config,
    validate_config_keys,
    create_pdf_from_ordered_images,
)
import os
import logging
import pytz
from trbdv0.logger_setup import setup_logger
from datetime import datetime
from trbdv0.send_email import (
    EmailSender,
    generate_subject_line,
    generate_email_body,
    get_attachments,
)
from trbdv0.constants import *
from trbdv0.survey_automation import send_survey, send_wearable_reminder
import argparse
import json
from trbdv0.survey_processor import init_processor, ISSProcessor
from lfp_analysis.lfp_dashboard import config_dash
import html


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run the notification system.")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to the config file"
    )
    args = parser.parse_args()

    config = read_config(args.config)

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
    patient_surveys = config["patient_surveys"]
    quatrics_config_path = config["quatrics_config_path"]
    quatrics_sleep_reminder = config["quatrics_sleep_reminder"]
    quatrics_nonwear_reminder = config["quatrics_nonwear_reminder"]
    timestamp = datetime.now(pytz.timezone(timezone))
    timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    today_date = get_todays_date()

    # load quatrics config
    quatrics_config = read_config(quatrics_config_path)

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
    all_free_responses = []  # To store all ISS free responses

    for patient in config["active_patients"]:
        # locate patient folder
        patient_in_dir = None
        for d in config["input_dir"]:
            potential_dir = os.path.join(d, patient, "oura")
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
        if warnings[SLEEP_VARIATION]:
            if patient in quatrics_sleep_reminder:
                logger.info(
                    f"{SLEEP_VARIATION} triggered, sending survey to {patient}..."
                )
                send_survey(patient, quatrics_config)

        # send quatrics survey if non_wear_time is triggered
        if warnings[LASTDAY_NON_WEAR_TIME_OVER_8]:
            if patient in quatrics_nonwear_reminder:
                logger.info(
                    f"{LASTDAY_NON_WEAR_TIME_OVER_8} triggered, sending survey to {patient}..."
                )
                send_wearable_reminder(patient, quatrics_config)

        # save summary stats to file
        summary_stats_file = os.path.join(patient_out_dir, f"{patient}.json")
        with open(summary_stats_file, "w") as f:
            json.dump(patient_summary_stats, f, indent=4)

        master.plot_combined_sleep_and_met()
        master.plot_daily_sleep_hours()

        pt_survey_strings = patient_surveys.get(patient, [])
        survey_plot_paths = []
        quatrics_results = {}
        for survey in pt_survey_strings:
            # check if survey class is defined
            class_path = SURVEY_CLASSES.get(survey)
            if not class_path:
                logger.warning(f"No survey processor class found for {survey}.")
                continue

            # check survey folder
            survey_folder = None
            for d in input_dir:  # `input_dir` is a list, so we iterate through it
                potential_survey_folder = os.path.join(d, patient, "qualtrics", survey)
                if os.path.exists(potential_survey_folder):
                    survey_folder = potential_survey_folder
                    logger.info(
                        f"Found survey folder for '{survey}' at: {survey_folder}"
                    )
                    break  # Exit the loop once the folder is found

            if not survey_folder:
                logger.warning(
                    f"Survey folder for '{survey}' does not exist for {patient} in any of the specified input directories."
                )
                continue

            # dynamically import the survey processor class
            processor = init_processor(
                SURVEY_CLASSES[survey], patient, survey_folder, patient_out_dir
            )

            # collect ISS free responses
            if isinstance(processor, ISSProcessor):
                logger.info(f"Checking for ISS free response for patient {patient}...")
                try:
                    free_response_data = processor.get_most_recent_free_response()
                    if free_response_data:
                        logger.info(
                            f"Found and stored ISS free response for {patient}."
                        )
                        all_free_responses.append(
                            {"patient": patient, **free_response_data}
                        )
                    else:
                        logger.info(f"No new ISS free response found for {patient}.")
                except Exception as e:
                    logger.error(f"Failed to get ISS free response for {patient}: {e}")

            quatrics_results[survey] = processor.get_latest_survey_results()
            if hasattr(processor, "plot_historical_scores"):
                survey_name = processor.survey_id
                safe_survey_name = survey_name.replace("-", "_").replace(" ", "_")
                logger.info(
                    f"Generating historical {survey_name} plot for {patient}..."
                )
                processor.plot_historical_scores(
                    output_filename=f"{patient}_{safe_survey_name}.png"
                )
                survey_plot_paths.append(
                    os.path.join(patient_out_dir, f"{patient}_{safe_survey_name}.png")
                )

        # Add LFP plot
        lfp_plot_path = os.path.join(patient_out_dir, f"{patient}_lfp.png")
        if os.path.exists(lfp_plot_path):
            logger.info(f"LFP plot already exists for {patient}, skipping generation.")
        else:
            # Generate LFP analysis plot
            logger.info(f"Generating LFP analysis plot for {patient}...")
            try:
                df_w_preds, fig = config_dash(patient, save_path=lfp_plot_path)
                if fig:
                    logger.info(f"LFP plot for {patient} created successfully.")
                else:
                    logger.warning(f"LFP plot could not be generated for {patient}.")

            except Exception as e:
                logger.error(
                    f"An error occurred during LFP plot generation for {patient}: {e}"
                )

        # Generate PDF report with ordered plots
        ordered_plots = []
        if os.path.exists(master.plot_save_path):
            ordered_plots.append(master.plot_save_path)
        if os.path.exists(master.sleep_hours_plot):
            ordered_plots.append(master.sleep_hours_plot)

        ordered_plots.extend(survey_plot_paths)

        lfp_plot_path = os.path.join(patient_out_dir, f"{patient}_lfp.png")
        if os.path.exists(lfp_plot_path):
            ordered_plots.append(lfp_plot_path)

        if ordered_plots:
            pdf_report_path = os.path.join(
                patient_out_dir, f"{patient}_{today_date}_report.pdf"
            )
            try:
                create_pdf_from_ordered_images(
                    patient, today_date, ordered_plots, pdf_report_path
                )
                all_attachments.append(pdf_report_path)
                logger.info(f"Successfully created simple PDF report for {patient}.")
            except Exception as e:
                logger.error(f"Failed to create PDF report for {patient}: {e}")
                logger.info("Attaching individual images as a fallback.")
                all_attachments.extend(ordered_plots)
        else:
            logger.warning(
                f"No images were generated for {patient}, so no report will be created."
            )

        all_patient_stats.append(
            {
                "summary": patient_summary_stats,
                "warning": warnings,
                "surveys": quatrics_results,
            }
        )

        logger.info(f"Data for patient {patient} processed successfully!")

    # send a single combined email for all ISS free responses
    if all_free_responses:
        logger.info("Consolidating all ISS free responses into a single secure email.")

        # Include patient IDs with responses in the subject line
        patient_ids_with_responses = sorted(
            [item["patient"] for item in all_free_responses]
        )
        patient_id_str = ", ".join(patient_ids_with_responses)
        secure_subject = f"SECURE: ISS Free Responses for Patient(s) {patient_id_str}"

        email_body_parts = [
            "<html><body style='font-family: sans-serif;'>"
            "<p>This is an automated, secure notification.</p>"
            "<p>The most recent free-text responses from ISS surveys have been retrieved for the following patient(s):</p>"
        ]

        for item in all_free_responses:
            patient_id = item["patient"]
            # Escape the response text to prevent any HTML characters in it from breaking the layout
            response_text = html.escape(item["response"] or "[No response entered]")
            response_date_str = item["date"]

            if response_date_str:
                response_date = datetime.strptime(
                    response_date_str, "%Y-%m-%d %H:%M:%S"
                ).strftime("%Y-%m-%d")
            else:
                response_date = "N/A"

            # Using HTML for better formatting in email clients.
            # Using inline styles for maximum compatibility.
            response_html = (
                f'<hr style="border: none; border-top: 1px solid #ccc; margin: 20px 0;">'
                f'<div style="font-size: 14px;">'
                f'<p style="margin: 0; padding: 0;"><b>Patient:</b> {patient_id}</p>'
                f'<p style="margin: 0; padding: 0;"><b>Response Date:</b> {response_date}</p>'
                f'<blockquote style="margin: 15px 0 0 20px; padding-left: 15px; border-left: 3px solid #eee; color: #333; font-style: italic;">'
                f'{response_text.replace(chr(10), "<br>")}'  # Replace newline characters with <br> tags
                f"</blockquote>"
                f"</div>"
            )
            email_body_parts.append(response_html)

        email_body_parts.append("</body></html>")
        secure_body = "".join(email_body_parts)

        try:
            email_sender.send_email(
                email_recipients, secure_subject, secure_body, attachments=[]
            )
            logger.info(
                "Consolidated secure ISS free response email sent successfully."
            )
        except Exception as e:
            logger.error(
                f"Failed to send consolidated secure ISS free response email: {e}"
            )

    # send the main daily report email
    if all_patient_stats:
        email_body = generate_email_body(all_patient_stats)
        subject = generate_subject_line(all_patient_stats)
        email_sender.send_email(email_recipients, subject, email_body, all_attachments)
        logger.info(
            f"Main daily report for patient(s) {config['active_patients']} sent successfully!"
        )
    else:
        logger.warning("No patient data was processed, skipping final report email.")

    email_sender.disconnect()


if __name__ == "__main__":
    main()
