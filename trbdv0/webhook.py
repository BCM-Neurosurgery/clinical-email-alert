"""Read the latest successfully staged Oura webhook timestamps."""

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from trbdv0.constants import (
    LATEST_ACTIVITY_WEBHOOK,
    LATEST_SLEEP_WEBHOOK,
)


_SUMMARY_KEY_BY_DATA_TYPE = {
    "sleep": LATEST_SLEEP_WEBHOOK,
    "daily_activity": LATEST_ACTIVITY_WEBHOOK,
}

# Delete events should not be presented as new available data.
_DATA_EVENT_TYPES = {"create", "update"}


def _iter_dicts(value: Any) -> Iterator[dict]:
    """Yield every dictionary in a nested JSON structure."""
    if isinstance(value, dict):
        yield value

        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _iter_dicts(child)

    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                yield from _iter_dicts(child)


def get_latest_webhook_times(patient_oura_dir: str, logger=None) -> dict:
    """Return the latest sleep and daily-activity webhook times.

    The patient directory is expected to have the structure:

        patient_oura_dir/YYYY-MM-DD/webhook_times.json

    Timestamps are compared in UTC and returned as ISO-8601 strings.
    Missing or unreadable data returns None and does not stop the report.
    """
    latest = {
        summary_key: None
        for summary_key in _SUMMARY_KEY_BY_DATA_TYPE.values()
    }

    webhook_paths = sorted(
        Path(patient_oura_dir).glob("*/webhook_times.json")
    )

    for path in webhook_paths:
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            if logger is not None:
                logger.warning(
                    f"Could not read webhook timestamps from {path}: {error}"
                )
            continue

        for event in _iter_dicts(payload):
            data_type = str(
                event.get("data_type", "")
            ).strip().casefold()

            summary_key = _SUMMARY_KEY_BY_DATA_TYPE.get(data_type)
            if summary_key is None:
                continue

            event_type = str(
                event.get("event_type", "")
            ).strip().casefold()

            if event_type not in _DATA_EVENT_TYPES:
                continue

            event_time = pd.to_datetime(
                event.get("event_time"),
                errors="coerce",
                utc=True,
            )

            if (
                not isinstance(event_time, pd.Timestamp)
                or pd.isna(event_time)
            ):
                if logger is not None:
                    logger.warning(
                        f"Invalid event_time in {path}; record skipped."
                    )
                continue

            current = latest[summary_key]
            if current is None or event_time > current:
                latest[summary_key] = event_time

    # Convert pandas timestamps to strings so the existing JSON summary
    # export in main.py continues to work.
    return {
        summary_key: (
            timestamp.isoformat()
            if timestamp is not None
            else None
        )
        for summary_key, timestamp in latest.items()
    }

def is_webhook_over_24(value, max_age_hours: int = 24) -> bool:
    """Return True when a valid webhook timestamp is over the age limit."""
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if (
        not isinstance(timestamp, pd.Timestamp)
        or pd.isna(timestamp)
    ):
        return False

    age = pd.Timestamp.now(tz="UTC") - timestamp
    return age > pd.Timedelta(hours=max_age_hours)