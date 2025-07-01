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
                with open(file, "rb") as attachment:
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
    yesterday_str = get_yesterdays_date()

    for entry in all_patient_stats:
        patient = entry["summary"]["patient"]

        # 1) any Wearables warning?
        has_oura_warn = any(entry["warning"].values())

        # 2) any Qualtrics warning from a survey filled *yesterday*?
        has_qual_warn = False
        for survey in entry.get("surveys", {}).values():
            enddate = survey.get("EndDate", "")
            date_part = enddate.split(" ")[0] if enddate else ""
            # only consider warnings if the survey was completed yesterday
            if date_part == yesterday_str and any(
                survey.get("latest_warnings", {}).values()
            ):
                has_qual_warn = True
                break

        if has_oura_warn or has_qual_warn:
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
    survey_rows = []

    def style(val, warnings_dict, *flags):
        if pd.isna(val) or any(warnings_dict.get(flag, False) for flag in flags):
            return f'<span style="background-color: #ff5252">{val}</span>'
        return val

    def format_score(sdict: dict, key: str, survey_name: str) -> str:
        """
        Format a single survey subscale score for inclusion in the HTML table.

        This will:
        1. Look up the raw score by `key` (e.g. "SC1", "SC0") in `sdict`.
        2. Check `sdict["latest_warnings"]` for any True flags.
        3. If any warning is True and `key` is in HIGHLIGHT_KEYS[survey_name], wrap the score
            in a red-background <span>.
        4. Append the survey's EndDate (date-only) in a lighter <small> tag on a new line.
        5. Return `np.nan` if the key is missing.

        Args:
            sdict (dict): Survey result dict containing at least:
                        - the score under `key`
                        - "EndDate" (a datetime string)
                        - "latest_warnings" (dict of str→bool)
            key (str): Subscale code, e.g. "SC1", "SC2", or "SC0".
            survey_name (str): One of "ISS", "PHQ-8", or "ASRM" to select highlight logic.

        Returns:
            str or float: An HTML string like "80<br><small>2025-06-11</small>",
                        possibly wrapped in <span style='background-color:#ff5252'>…</span>;
                        or `np.nan` if `sdict.get(key)` is None.
        """
        raw = sdict.get(key)
        if raw is None:
            return np.nan

        warn_dict = sdict.get("latest_warnings", {})
        # highlight only if any warning and this subscale is in our map
        if any(warn_dict.values()) and key in HIGHLIGHT_KEYS.get(survey_name, []):
            raw = f"<span style='background-color:#ff5252'>{raw}</span>"

        ed = sdict.get("EndDate", "")
        if ed:
            date_str = ed.split()[0]
            return f"{raw}<br><small style='color:#888'>{date_str}</small>"

        return raw

    for entry in all_patient_stats:
        summary = entry["summary"]
        warnings = entry["warning"]
        patient = summary["patient"]
        surveys = entry.get("surveys", {})

        df_rows.append(
            {
                PT_COLUMN: patient,
                LASTDAY_SLEEP_COLUMN: style(
                    f"{summary.get(LASTDAY_SLEEP_HOURS, np.nan):.1f}",
                    warnings,
                    LASTDAY_SLEEP_NAN,
                    LASTDAY_SLEEP_LESS_THAN_6,
                    SLEEP_VARIATION,
                ),
                AVERAGE_SLEEP_COLUMN: style(
                    f"{summary.get(AVERAGE_SLEEP_HOURS, np.nan):.1f}",
                    warnings,
                    AVERAGE_SLEEP_NAN,
                ),
                LASTDAY_STEPS_COLUMN: style(
                    f"{summary.get(LASTDAY_STEPS, np.nan):.0f}",
                    warnings,
                    # LASTDAY_STEPS_NAN,
                    # STEPS_VARIATION,
                ),
                AVERAGE_STEPS_COLUMN: style(
                    f"{summary.get(AVERAGE_STEPS, np.nan):.0f}",
                    warnings,
                    # AVERAGE_STEPS_NAN,
                ),
                LASTDAY_MET_COLUMN: style(
                    f"{summary.get(LASTDAY_MET, np.nan):.2f}",
                    warnings,
                    LASTDAY_MET_NAN,
                    MET_VARIATION,
                ),
                AVERAGE_MET_COLUMN: style(
                    f"{summary.get(AVERAGE_MET, np.nan):.2f}",
                    warnings,
                    AVERAGE_MET_NAN,
                ),
            }
        )

        # Survey scores row
        iss = surveys.get("ISS", {})
        phq8 = surveys.get("PHQ-8", {})
        asrm = surveys.get("ASRM", {})

        survey_rows.append(
            {
                "Patient": patient,
                "Activation (ISS)": format_score(iss, "SC1", "ISS"),
                "Well-being (ISS)": format_score(iss, "SC2", "ISS"),
                "Perceived Conflict (ISS)": format_score(iss, "SC3", "ISS"),
                "Depression Index (ISS)": format_score(iss, "SC4", "ISS"),
                "Depression Score (PHQ-8)": format_score(phq8, "SC0", "PHQ-8"),
                "Mania Score (ASRM)": format_score(asrm, "SC0", "ASRM"),
            }
        )

    df_summary = pd.DataFrame(df_rows)

    html_summary_table = "<h3>Daily Summary from Wearables</h3>" + df_summary.to_html(
        index=False, escape=False, na_rep="nan"
    )

    # Determine if ANY patient is in the allowed list
    patients = {entry["summary"]["patient"] for entry in all_patient_stats}
    include_surveys = bool(patients & ALLOWED_SURVEY_PATIENTS)

    html_survey_table = ""
    if include_surveys:
        # … [build survey_rows exactly as before] …
        df_survey = pd.DataFrame(survey_rows)
        html_survey_table = (
            "<h3>Most Recent Qualtrics Survey Scores</h3>"
            + df_survey.to_html(index=False, escape=False, na_rep="nan")
        )

    lastday = get_last_day()
    yesterday = get_yesterdays_date()
    today = get_todays_date()

    note = (
        f"<p style='font-size: 0.9em; color: #555;'>"
        f"<strong>Note:</strong> Cells highlighted in red indicate "
        f"either <em>missing data</em>, sleep less than 6 hours, or values that deviate more than ±25% from the patient's average.<br>"
        f"<strong>Sleep (12pm-12pm, Day-2 to Yesterday)</strong> is calculated as the sleep duration from "
        f"<em>12:00 PM on {lastday}</em> to <em>12:00 PM on {yesterday}</em>.<br>"
        f"<strong>Day-2 Steps</strong> is calculated as the step counts from "
        f"<em>4:00 AM on {lastday}</em> to <em>4:00 AM on {yesterday}</em>.<br>"
        f"<strong>Day-2 Average MET</strong> is calculated as the average MET score from "
        f"<em>4:00 AM on {lastday}</em> to <em>4:00 AM on {yesterday}</em>."
        f"</p>"
    )

    # 4) Survey scoring legend + ISS mood‐state thresholds
    survey_note = ""
    if include_surveys:
        survey_note = (
            "<p style='font-size:0.9em; color:#555;'>"
            "<strong>Survey Score Thresholds:</strong><br>"
            "• <strong>Activation ≥ 155 &amp; Well-Being ≥ 125</strong> -> (Hypo)mania<br>"
            "• <strong>Activation ≥ 155 &amp; Well-Being &lt; 125</strong> -> Mixed State<br>"
            "• <strong>Activation &lt; 155 &amp; Well-Being ≥ 125</strong> -> Euthymia<br>"
            "• <strong>Activation &lt; 155 &amp; Well-Being &lt; 125</strong> -> Depression<br>"
            "• <strong>PHQ-8 (SC0) &gt; 10</strong> → Depression<br>"
            "• <strong>ASRM (SC0) &gt; 6</strong> → (Hypo)mania"
            "</p>"
        )

    return html_summary_table + note + html_survey_table + survey_note


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
