import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from trbdv0.master import Master


class MasterDSTTests(unittest.TestCase):
    def setUp(self):
        self.master = Master.__new__(Master)
        self.master.timezone = "America/Chicago"
        self.master.logger = Mock()
        self.master.sleep = SimpleNamespace(sleep_data=[])
        self.master.activity = SimpleNamespace(activity_phases=[], activity_data=[])

    def assert_timestamps(self, frame, expected):
        self.assertEqual(
            [timestamp.isoformat() for timestamp in frame["timestamp"]], expected
        )
        self.assertFalse(frame["timestamp"].duplicated().any())
        self.assertEqual(str(frame["timestamp"].dt.tz), "America/Chicago")

    def test_bedtime_rounding_across_dst_transitions(self):
        cases = [
            (
                "2026-10-31", "01:10:42-05:00", "01:11:01-05:00",
                "01:10:00-05:00", "01:11:00-05:00",
            ),
            (
                "2026-11-02", "01:10:42-06:00", "01:11:01-06:00",
                "01:10:00-06:00", "01:11:00-06:00",
            ),
            (
                "2026-11-01", "01:10:42-05:00", "01:11:01-05:00",
                "01:10:00-05:00", "01:11:00-05:00",
            ),
            (
                "2026-11-01", "01:10:42-06:00", "01:11:01-06:00",
                "01:10:00-06:00", "01:11:00-06:00",
            ),
            (
                "2026-11-01", "00:59:42-05:00", "01:00:42-05:00",
                "00:59:00-05:00", "01:00:00-05:00",
            ),
            (
                "2026-11-01", "01:58:42-05:00", "01:59:42-05:00",
                "01:58:00-05:00", "01:59:00-05:00",
            ),
            (
                "2026-03-08", "01:58:42-06:00", "01:59:42-06:00",
                "01:58:00-06:00", "01:59:00-06:00",
            ),
        ]
        for day, start, end, first_minute, last_minute in cases:
            with self.subTest(day=day, start=start, end=end):
                bedtimes = [
                    {"bedtime_start": f"{day}T{start}", "bedtime_end": f"{day}T{end}"}
                ]
                self.master.sleep.get_bedtimes = lambda: bedtimes
                self.assert_timestamps(
                    self.master.build_master_sleep_bedtimes(),
                    [f"{day}T{first_minute}", f"{day}T{last_minute}"],
                )

    def test_bedtimes_keep_both_repeated_hours(self):
        bedtimes = [
            {
                "bedtime_start": f"2026-11-01T01:10:42{offset}",
                "bedtime_end": f"2026-11-01T01:11:01{offset}",
            }
            for offset in ("-05:00", "-06:00")
        ]
        self.master.sleep.get_bedtimes = lambda: bedtimes
        self.assert_timestamps(
            self.master.build_master_sleep_bedtimes(),
            [
                f"2026-11-01T01:{minute}:00{offset}"
                for offset in ("-05:00", "-06:00")
                for minute in (10, 11)
            ],
        )

    def test_sleep_phases_keep_both_repeated_hours(self):
        self.master.sleep.sleep_data = [
            {"bedtime_start": "2026-11-01T01:10:42-05:00", "sleep_phase_5_min": "1"},
            {"bedtime_start": "2026-11-01T01:10:42-06:00", "sleep_phase_5_min": "2"},
        ]
        frame = self.master.build_master_sleep_phases()
        self.assert_timestamps(frame, self.repeated_five_minutes())
        self.assertEqual(frame["phase_label"].tolist(), ["deep"] * 5 + ["light"] * 5)
        self.master.logger.warning.assert_not_called()

    def test_activity_phases_keep_both_repeated_hours(self):
        self.master.activity.activity_phases = [
            {"timestamp": "2026-11-01T01:10:42-05:00", "class_5_min": "1"},
            {"timestamp": "2026-11-01T01:10:42-06:00", "class_5_min": "3"},
        ]
        frame = self.master.build_master_activity_phase()
        self.assert_timestamps(frame, self.repeated_five_minutes())
        self.assertEqual(
            frame["activity_label"].tolist(), ["rest"] * 5 + ["low_activity"] * 5
        )
        self.master.logger.warning.assert_not_called()

    def test_met_keeps_both_repeated_hours(self):
        self.master.activity.activity_data = [
            {
                "met": {
                    "timestamp": "2026-11-01T01:10:42-05:00",
                    "interval": 60,
                    "items": [1.1, 1.2],
                }
            },
            {
                "met": {
                    "timestamp": "2026-11-01T01:10:42-06:00",
                    "interval": 60,
                    "items": [2.1, 2.2],
                }
            },
        ]
        frame = self.master.build_master_met()
        self.assert_timestamps(
            frame,
            [
                f"2026-11-01T01:{minute}:00{offset}"
                for offset in ("-05:00", "-06:00")
                for minute in (10, 11)
            ],
        )
        self.assertEqual(frame["met"].tolist(), [1.1, 1.2, 2.1, 2.2])
        self.master.logger.warning.assert_not_called()

    @staticmethod
    def repeated_five_minutes():
        return [
            f"2026-11-01T01:{minute}:00{offset}"
            for offset in ("-05:00", "-06:00")
            for minute in range(10, 15)
        ]


if __name__ == "__main__":
    unittest.main()
