from datetime import datetime, timedelta
import json
import os
import numpy as np
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


def get_todays_date(timezone: str) -> str:
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


def get_yesterdays_date(timezone: str) -> str:
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


def calculate_average_met(activity_data: dict) -> float:
    """
    Calculate the average MET value for periods when the ring was worn.

    Parameters:
    activity_data (dict): A dictionary containing activity classification ('class_5_min')
                          and MET ('met') data. E.g. {'day': '2025-02-06',
                          'sleep_phase_5_min': '444222211111233322211111111222333332224444222222222123333333222222422334222222',
                          'bedtime_start': '2025-02-05T23:45:02-05:00', 'bedtime_end': '2025-02-06T06:14:25-05:00',
                          'class_5_min': '1111111111111111111111111133221222322200000000000000000000000000333300032222222334334...',
                          'non_wear_time': 15600, 'timestamp': '2025-02-06T04:00:00-05:00',
                          'steps': 6674, 'met': {'interval': 60.0, 'items': [...], 'timestamp': '2025-02-06T04:00:00.000-05:00'}}

    Returns:
    float or None: The average MET value for periods when the ring was worn,
                   or None if no valid MET values exist.
    """
    class_5_min = np.array([int(c) for c in activity_data["class_5_min"]])
    met_items = np.array(
        [float(m) if m is not None else np.nan for m in activity_data["met"]["items"]],
        dtype=float,
    )

    interval_seconds = int(activity_data["met"]["interval"])
    samples_per_class = 300 // interval_seconds

    # Expand class_5_min values to match MET sampling rate
    expanded_class = np.repeat(class_5_min, samples_per_class)

    # Ensure the expanded class and MET items have the same length
    min_length = min(len(expanded_class), len(met_items))
    expanded_class = expanded_class[:min_length]
    met_items = met_items[:min_length]

    valid_met_items = met_items[expanded_class != 0]

    return np.nanmean(valid_met_items) if valid_met_items.size > 0 else None
