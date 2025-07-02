from datetime import datetime, timedelta
import json
import pytz
from fpdf import FPDF
import os

PHASE_MAPPING = {
    "1": "Deep Sleep",
    "2": "Light Sleep",
    "3": "REM Sleep",
    "4": "Awake",
    "non_wear_time": "Non-Wear Time",
    "steps": "Step Count",
    "average_met": "Average MET",
    "met_interval": "MET Interval",
    "met_items": "MET Items",
    "met_timestamp": "MET Timestamp",
    "class_5_mins": "Activity Classification",
}


# Helper function to parse the date and time
def parse_date(date_str):
    """
    E.g. date_str '2019-12-04'
    returns datetime.date(2019, 12, 4)
    """
    # Reformat to ignore timezone
    return datetime.fromisoformat(date_str).replace(tzinfo=None)


def format_seconds(seconds: int) -> str:
    """
    convert elapsed such as 1230 -> 00:00:00 format
    """
    # Create a timedelta object based on the number of seconds
    td = timedelta(seconds=seconds)
    # Format the hours, minutes, and seconds as a string
    return str(td)


def get_todays_date(timezone="America/Chicago") -> str:
    """
    Get today's date in the specified timezone.

    Args:
        timezone (str): IANA timezone string, e.g. "America/Chicago"

    Returns:
        str: Date string in "YYYY-MM-DD" format, e.g. "2024-05-01"
    """
    tz = pytz.timezone(timezone)
    now_local = datetime.now(tz)
    return now_local.strftime("%Y-%m-%d")


def get_yesterdays_date(timezone="America/Chicago") -> str:
    """
    Get the date of yesterday in the specified timezone.

    Args:
        timezone (str): IANA timezone string, e.g. "America/Chicago"

    Returns:
        str: Date string in "YYYY-MM-DD" format, e.g. "2024-04-30"
    """
    tz = pytz.timezone(timezone)
    now_local = datetime.now(tz)
    yesterday_local = now_local - timedelta(days=1)
    return yesterday_local.strftime("%Y-%m-%d")


def get_last_day(timezone="America/Chicago") -> str:
    """
    Get the date of the day before yesterday in the specified timezone.

    Args:
        timezone (str): IANA timezone string, e.g. "America/Chicago"

    Returns:
        str: Date string in "YYYY-MM-DD" format, e.g. "2024-04-29"
    """
    tz = pytz.timezone(timezone)
    now_local = datetime.now(tz)
    day_before_yesterday_local = now_local - timedelta(days=2)
    return day_before_yesterday_local.strftime("%Y-%m-%d")


def get_past_dates(end_date: str, past_days: int = 7) -> list:
    """Get a sorted list of past dates before the given end_date.

    Args:
        end_date (str): The end date in "YYYY-MM-DD" format (excluded from result).
        past_days (int): Number of past days to retrieve before the end_date. Default is 7.

    Returns:
        list: A list of dates (as strings) in "YYYY-MM-DD" format,
              sorted from the earliest to latest, excluding the end_date itself.
    """
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [end_date_dt - timedelta(days=x) for x in range(1, past_days + 1)]
    sorted_dates = sorted(date_list)
    return [date.strftime("%Y-%m-%d") for date in sorted_dates]


def get_iter_dates(end_date: str, past_days: int = 7) -> list:
    """
    Get a list of dates including both the start and end date.

    Args:
        end_date (str): The end date in "YYYY-MM-DD" format (included in result).
        past_days (int): Number of days before the end_date to include (inclusive of end_date). Default is 7.

    Returns:
        list: A list of dates (as strings) in "YYYY-MM-DD" format,
              sorted from earliest to latest, inclusive of both start and end date.
    """
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [end_date_dt - timedelta(days=x) for x in reversed(range(past_days))]
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


def validate_config_keys(config: dict) -> None:
    """
    Validate that the required keys are present in the configuration dictionary.

    Args:
        config (dict): The loaded JSON configuration dictionary.

    Raises:
        KeyError: If any required key is missing from the configuration.

    Example:
        >>> with open("config.json") as f:
        ...     config = json.load(f)
        >>> validate_config_keys(config)
    """
    required_keys = [
        "input_dir",
        "output_dir",
        "log_dir",
        "timezone",
        "email_recipients",
        "secure_email_recipients",
        "smtp_server",
        "smtp_port",
        "smtp_user",
        "smtp_password",
        "past_days",
        "active_patients",
        "quatrics_config_path",
        "quatrics_sleep_reminder",
        "quatrics_nonwear_reminder",
        "study_name",
    ]

    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise KeyError(f"Missing required config keys: {', '.join(missing_keys)}")


def create_pdf_from_ordered_images(
    patient_id: str, report_date: str, ordered_image_files: list, output_path: str
):
    """
    Creates a multi-page PDF from an ordered list of images.

    If an image filename contains "sleep_hours", it is placed on the same
    page as the image that came before it. The function automatically
    resizes the image pair to fit on one page.

    Args:
        patient_id (str): The ID of the patient for the report title.
        report_date (str): The date of the report for the header.
        ordered_image_files (list): An ORDERED list of full paths to the image files.
                                    A "sleep_hours" image should directly follow the
                                    "wearables" image it belongs to.
        output_path (str): The full path to save the output PDF file.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- PDF Layout Constants (in mm) ---
    A4_WIDTH, A4_HEIGHT = 210, 297
    MARGIN = 15
    USABLE_WIDTH = A4_WIDTH - 2 * MARGIN
    USABLE_HEIGHT = A4_HEIGHT - 2 * MARGIN
    HEADER_HEIGHT = 20  # Approximate height for title + line break
    SPACE_BETWEEN_IMAGES = 5

    # Use a while loop to have more control over the iteration
    i = 0
    while i < len(ordered_image_files):
        main_image_file = ordered_image_files[i]

        if not os.path.exists(main_image_file):
            print(f"[Warning] Image file not found, skipping: {main_image_file}")
            i += 1
            continue

        pdf.add_page()

        # --- Add a consistent header to each page ---
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"Patient Report: {patient_id} - {report_date}", 0, 1, "C")
        pdf.ln(10)  # Part of HEADER_HEIGHT

        # --- Look ahead to see if the next image is a sleep plot ---
        is_paired = (
            i + 1 < len(ordered_image_files)
            and "sleep_hours" in ordered_image_files[i + 1]
        )

        if is_paired:
            # --- Two-Image Page Logic ---
            sleep_image_file = ordered_image_files[i + 1]
            if not os.path.exists(sleep_image_file):
                print(f"[Warning] Paired image not found, skipping: {sleep_image_file}")
                # Fallback to printing just the main image
                pdf.image(main_image_file, w=USABLE_WIDTH)
                i += 1  # Only advance by one
                continue

            # Calculate height for each image to fit on the page
            available_height = USABLE_HEIGHT - HEADER_HEIGHT - SPACE_BETWEEN_IMAGES
            max_height_per_image = available_height / 2

            # Add the first (wearables) image
            pdf.image(main_image_file, w=USABLE_WIDTH, h=max_height_per_image)
            pdf.ln(SPACE_BETWEEN_IMAGES)

            # Add the second (sleep_hours) image
            pdf.image(sleep_image_file, w=USABLE_WIDTH, h=max_height_per_image)

            # Advance the loop by 2 since we processed a pair of images
            i += 2
        else:
            # --- Single-Image Page Logic ---
            pdf.image(main_image_file, w=USABLE_WIDTH)
            # Advance the loop by 1
            i += 1

    pdf.output(output_path)
    print(f"PDF report with paired images generated at: {output_path}")
