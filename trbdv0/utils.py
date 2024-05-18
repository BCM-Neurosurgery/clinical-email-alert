from datetime import datetime, timedelta


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
