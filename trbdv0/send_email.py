import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List
import os
import pandas as pd
import numpy as np
from trbdv0.utils import (
    get_todays_date,
    get_yesterdays_date,
    get_last_day,
)
from trbdv0.constants import *


class EmailSender:
    def __init__(self, smtp_server, smtp_port, smtp_user, smtp_password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.server = None

    def connect(self):
        self.server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        self.server.ehlo()
        self.server.starttls()
        self.server.login(self.smtp_user, self.smtp_password)

    def send_email(
        self,
        to_addrs: List[str],
        subject: str,
        body: str,
        attachments: List[str] = None,
    ):
        msg = MIMEMultipart()
        msg["From"] = self.smtp_user
        msg["To"] = ", ".join(to_addrs)
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "html"))

        if attachments:
            for file in attachments:
                attachment = open(file, "rb")
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {os.path.basename(file)}",
                )
                msg.attach(part)

        try:
            self.server.sendmail(self.smtp_user, to_addrs, msg.as_string())
            return "Email sent successfully with attachments"
        except Exception as e:
            return f"Failed to send email: {str(e)}"

    def disconnect(self):
        if self.server:
            self.server.quit()


def generate_subject_line(all_patient_stats: list) -> str:
    """
    Generate a concise subject line for doctors, listing patients needing attention,
    and briefly noting all-clear patients.

    Args:
        all_patient_stats (list): list of dicts

    Returns:
        str: Subject line
    """
    needs_attention = []
    all_clear = []

    # all belong to the same study name for now
    # different than study ids on Elias, only
    # for email showing purpose
    study_name = all_patient_stats[0]["summary"]["study_name"]

    for entry in all_patient_stats:
        patient = entry["summary"]["patient"]
        warnings = entry["warning"]

        if any(warnings.values()):
            needs_attention.append(patient)
        else:
            all_clear.append(patient)

    if not needs_attention:
        patients_str = ", ".join(sorted(all_clear))
        return f"{study_name} [All Clear] for Patients: {patients_str}"

    flagged_str = ", ".join(sorted(needs_attention))
    return f"{study_name} [Warning: {flagged_str} need review]"


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
            PT_COLUMN: patient,
            LASTDAY_SLEEP_COLUMN: style(
                f"{summary.get(LASTDAY_SLEEP_HOURS, np.nan):.1f}",
                LASTDAY_SLEEP_NAN,
                LASTDAY_SLEEP_LESS_THAN_6,
                SLEEP_VARIATION,
            ),
            AVERAGE_SLEEP_COLUMN: style(
                f"{summary.get(AVERAGE_SLEEP_HOURS, np.nan):.1f}",
                AVERAGE_SLEEP_NAN,
            ),
            LASTDAY_STEPS_COLUMN: style(
                f"{summary.get(LASTDAY_STEPS, np.nan):.0f}",
                # LASTDAY_STEPS_NAN,
                # STEPS_VARIATION,
            ),
            AVERAGE_STEPS_COLUMN: style(
                f"{summary.get(AVERAGE_STEPS, np.nan):.0f}",
                # AVERAGE_STEPS_NAN,
            ),
            LASTDAY_MET_COLUMN: style(
                f"{summary.get(LASTDAY_MET, np.nan):.2f}",
                LASTDAY_MET_NAN,
                MET_VARIATION,
            ),
            AVERAGE_MET_COLUMN: style(
                f"{summary.get(AVERAGE_MET, np.nan):.2f}",
                AVERAGE_MET_NAN,
            ),
        }

        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    html_table = df.to_html(index=False, escape=False)
    lastday = get_last_day()
    yesterday = get_yesterdays_date()

    note = (
        f"<p style='font-size: 0.9em; color: #555;'>"
        f"<strong>Note:</strong> Cells highlighted in red indicate "
        f"either <em>missing data</em>, sleep less than 6 hours, or values that deviate more than ±25% from the patient's average.<br>"
        f"<strong>Last Day</strong> is defined as the 24-hour period from "
        f"<em>12:00 PM on {lastday}</em> to <em>12:00 PM on {yesterday}</em>."
        f"</p>"
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
