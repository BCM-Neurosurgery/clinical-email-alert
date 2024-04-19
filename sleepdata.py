import json
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
import utils
import pandas as pd
from collections import Counter
import numpy as np

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

    def plot_sleep_interval_on_day(self, ax, day: str) -> None:
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

        # Plotting
        if ax is None:
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
        ax.set_xlim([all_times[0], all_times[-1]])
        ax.set_ylim(0.5, 1.5)
        ax.set_title(f"Sleep Intervals for {day}")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        # by default it plots in UTC time
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        # Improve layout and show grid
        plt.grid(True)

    def get_summary_stat_for_day(self, day: str) -> pd.DataFrame:
        """
        Get a summary of stats for day
        each row represents a sleep section during the same day
        headers include
            - bedtime_start
            - bedtime_end
            - time_in_bed
            - awake_time
            - deep_sleep_duration
            - total_sleep_duration
        format the time such that they are in 00:00:00 format
        """
        sleep_data_on_day = self.get_sleep_data_on_day(day)
        temp = []
        for sleep_chunk in sleep_data_on_day:
            day_chunk = {}
            day_chunk["bedtime_start"] = sleep_chunk["bedtime_start"]
            day_chunk["bedtime_end"] = sleep_chunk["bedtime_end"]
            day_chunk["time_in_bed"] = utils.format_seconds(sleep_chunk["time_in_bed"])
            day_chunk["awake_time"] = utils.format_seconds(sleep_chunk["awake_time"])
            day_chunk["light_sleep_duration"] = utils.format_seconds(
                sleep_chunk["light_sleep_duration"]
            )
            day_chunk["deep_sleep_duration"] = utils.format_seconds(
                sleep_chunk["deep_sleep_duration"]
            )
            day_chunk["total_sleep_duration"] = utils.format_seconds(
                sleep_chunk["total_sleep_duration"]
            )
            temp.append(day_chunk)
        return pd.DataFrame.from_records(temp)

    def get_summary_plot_for_day(self, day: str):
        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Call the first function and pass its ax to the first subplot
        self.plot_sleep_interval_on_day(axes[0], day)
        self.plot_sleep_phase_5_min(axes[1], day)
        self.plot_sleep_distribution_for_day(None, day)

        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()

    def plot_sleep_phase_5_min(self, ax, day):
        sleep_data_on_day = self.get_sleep_data_on_day(day)

        if ax is None:
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
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend()
        plt.grid(True)

    def plot_sleep_distribution_for_day(self, ax, day):
        """
        Plot the distribution of sleep phase per day
        """
        # Initialize a dictionary to hold percentage data for each sample
        percentage_data = []

        sleep_data_on_day = self.get_sleep_data_on_day(day)

        # Process each item in data
        for item in sleep_data_on_day:
            sleep_data = item["sleep_phase_5_min"]
            count = Counter(sleep_data)
            total = sum(count.values())
            percentages = {phase: (cnt / total * 100) for phase, cnt in count.items()}
            percentage_data.append(percentages)

        phases = ["1", "2", "3", "4"]

        # Prepare data for plotting
        plot_data = {phase: [] for phase in phases}
        for pdata in percentage_data:
            for phase in phases:
                plot_data[phase].append(pdata.get(phase, 0))

        # Plotting
        fig, ax = plt.subplots()
        # Location of bars on the x-axis
        bar_width = 0.2
        x_positions = np.arange(len(sleep_data_on_day))

        # Plot each phase as separate bars in a group
        for i, phase in enumerate(phases):
            bar_positions = x_positions + i * bar_width
            bars = ax.bar(
                bar_positions, plot_data[phase], width=bar_width, label=f"Phase {phase}"
            )

            # Adding percentage labels above bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        ax.set_ylabel("Percentage")
        ax.set_title("Percentage Distribution of Sleep Phases per Entry")
        ax.set_xticks(x_positions + bar_width * (len(phases) - 1) / 2)
        ax.set_xticklabels([f"Section {i+1}" for i in x_positions])
        ax.legend()


class SleepDataOneDay:
    def __init__(self) -> None:
        pass
