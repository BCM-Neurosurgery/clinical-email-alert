from datetime import datetime, timedelta
import json
import pytz


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


def get_day_before_yesterday(timezone="America/Chicago") -> str:
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
        "smtp_server",
        "smtp_port",
        "smtp_user",
        "smtp_password",
        "past_days",
        "active_patients",
        "quatrics_sleep_reminder",
        "quatrics_nonwear_reminder",
        "study_name",
    ]

    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise KeyError(f"Missing required config keys: {', '.join(missing_keys)}")
