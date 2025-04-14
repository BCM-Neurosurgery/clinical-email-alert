import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
import pytz


class CSTFormatter(logging.Formatter):
    """Logging Formatter to add timestamps in the US Central timezone."""

    def __init__(self, fmt=None, datefmt=None, tz="America/Chicago"):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.tz = pytz.timezone(tz)

    def converter(self, timestamp):
        # New: Use timezone-aware UTC object and convert to target timezone
        dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt_utc.astimezone(self.tz)

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat()


def setup_logger(name, file_path, tz, level=logging.INFO):
    """Setup logger with custom timezone formatter."""
    formatter = CSTFormatter("%(asctime)s - %(levelname)s - %(message)s", tz=tz)
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
