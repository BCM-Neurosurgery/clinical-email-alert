from datetime import datetime


# Helper function to parse the date and time
def parse_date(date_str):
    """
    E.g. date_str '2019-12-04'
    returns datetime.date(2019, 12, 4)
    """
    parsed = datetime.fromisoformat(date_str)
    # Reformat to ignore timezone
    dt_without_tz = parsed.replace(tzinfo=None)
    return dt_without_tz
