import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pytz
import utils

# TODO: how to define a day?


class SleepData:
    def __init__(self, path) -> None:
        """
        Args:
            path (str): path to json
        """
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.jsonObj = json.load(f)

        # self.data looks like [{}, {}, ..., {}]
        self.data = self.jsonObj["data"]
        # self.sleep_data_one_day -> SleepDataOneDay object
        self.sleep_data_one_day = {}

    def get_available_days(self) -> set:
        """
        Return all recorded days
        """
        res = set()
        for sleep_chunk in self.data:
            res.add(sleep_chunk["day"])
        return res

    def get_sleep_data_on_day(self, day: str) -> list:
        """
        Return a list of sleep data on that day
        Args:
            day: looks like 2023-09-14
        """
        res = []
        for sleep_chunk in self.data:
            if sleep_chunk["day"] == day:
                res.append(sleep_chunk)
        return res

    def plot_sleep_interval_on_day(self, day: str) -> None:
        """
        Plot sleep intervals on day
        """
        sleep_data_on_day = self.get_sleep_data_on_day(day)

        # Create lists for sleep start and end times
        sleep_start_times = []
        sleep_end_times = []
        all_times = []

        for entry in sleep_data_on_day:
            start_time = utils.parse_date(entry["bedtime_start"])
            end_time = utils.parse_date(entry["bedtime_end"])
            sleep_start_times.append(start_time)
            sleep_end_times.append(end_time)
            all_times.append(start_time)
            all_times.append(end_time)

        # sort
        all_times.sort()
        print(all_times)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 3))

        # Add each sleep interval to the plot
        for start, end in zip(sleep_start_times, sleep_end_times):
            ax.plot(
                [start, end],
                [1, 1],
                marker="|",
                markersize=25,
                linewidth=1,
                color="blue",
            )

        # Set labels and title
        ax.set_yticks([])
        # ax.set_xlim([datetime(2023, 9, 14), datetime(2023, 9, 15)])
        ax.set_xlim([all_times[0], all_times[-1]])
        ax.set_ylim(0.5, 1.5)
        ax.set_title(f"Sleep Intervals for {day}")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        # by default it plots in UTC time
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # Improve layout and show grid
        plt.grid(True)
        plt.gcf().autofmt_xdate()

        plt.show()
        return ax

    def get_summary_for_day(self, day: str) -> dict:
        """
        Get a summary of stats for day
        including
        1. number of hours of sleep
        2.
        """
        sleep_data_on_day = self.get_sleep_data_on_day(day)
        res = {}
        res["total_awake_time"] = 0
        res["total_deep_sleep_duration"] = 0
        res["total_sleep_duration"] = 0
        res["total_time_in_bed"] = 0
        for sleep_chunk in sleep_data_on_day:
            res["total_awake_time"] += sleep_chunk["awake_time"]
            res["total_deep_sleep_duration"] += sleep_chunk["deep_sleep_duration"]
            res["total_light_sleep_duration"] += sleep_chunk["light_sleep_duration"]
            res["total_time_in_bed"] += sleep_chunk["time_in_bed"]
        return res

    def plot_sleep_phase_5_min(self, day):
        sleep_data_on_day = self.get_sleep_data_on_day(day)

        fig, ax = plt.subplots()

        # Process each sleep session
        for session in sleep_data_on_day:
            start_time = utils.parse_date(session["bedtime_start"])
            # end_time = datetime.fromisoformat(session['bedtime_end'])
            sleep_phases = session["sleep_phase_5_min"]

            # Generate time series for the session
            times = [
                start_time + timedelta(minutes=5 * i) for i in range(len(sleep_phases))
            ]
            values = [int(char) for char in sleep_phases]

            # Plotting
            ax.plot(
                times,
                values,
                marker="o",
                linestyle="-",
                label=f"Session starting {start_time.strftime('%Y-%m-%d %H:%M')}",
            )

        # Formatting the plot
        ax.set_title("Sleep Phase Values Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Sleep Phase Value")
        # ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend()

        plt.gcf().autofmt_xdate()  # Auto formats the date labels
        plt.grid(True)
        plt.show()


class SleepDataOneDay:
    def __init__(self) -> None:
        pass
