import json
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, time
import matplotlib.dates as mdates
import utils
import pandas as pd
from collections import Counter
import numpy as np

# TODO: how to define a day?


class SleepData:
    def __init__(self, data) -> None:
        """
        Args:
            path (str): path to json
        """
        # self.data looks like [{}, {}, ..., {}]
        self.data = data
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
        self.plot_sleep_distribution_for_day(day)

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

    def plot_sleep_distribution_for_day(self, day):
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

        # Define phase mapping
        phase_mapping = {
            "1": "Deep Sleep",
            "2": "Light Sleep",
            "3": "REM Sleep",
            "4": "Awake",
        }

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
                bar_positions,
                plot_data[phase],
                width=bar_width,
                label=f"{phase_mapping[phase]}",
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

    def plot_sleep_distribution_for_week(self):
        """
        Plot sleep distribution for the past week ending at day
        """

        sleep_data = {}

        for entry in self.data:
            day = entry["day"]
            sleep_phases = entry["sleep_phase_5_min"]
            phase_counts = Counter(sleep_phases)  # Count occurrences of each phase

            if day not in sleep_data:
                sleep_data[day] = Counter()

            sleep_data[day] += phase_counts

        # Convert counts to hours (assuming each phase count is in 5 minutes increments)
        for day in sleep_data:
            for phase in sleep_data[day]:
                sleep_data[day][phase] = (
                    sleep_data[day][phase] * 5
                ) / 60  # Convert to hours

        # Creating DataFrame
        df = pd.DataFrame.from_dict(sleep_data, orient="index").fillna(0)
        # Rename columns based on sleep phase descriptions
        phase_mapping = {
            "1": "Deep Sleep",
            "2": "Light Sleep",
            "3": "REM Sleep",
            "4": "Awake",
        }
        df.columns = [phase_mapping.get(col, f"Phase {col}") for col in df.columns]

        # reorder so that awake is at the top
        column_order = ["Deep Sleep", "Light Sleep", "REM Sleep", "Awake"]
        df = df[column_order]

        # Plotting
        ax = df.plot(kind="bar", stacked=True, figsize=(10, 7))
        ax.set_title("Distribution of Sleep Phases per Day")
        ax.set_ylabel("Hours")
        plt.xticks(rotation=45)

        # Adding labels to each bar
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            if (
                height > 0
            ):  # Only label the bar if the height is significant to avoid clutter
                x, y = p.get_xy()
                ax.text(
                    x + width / 2,
                    y + height / 2,
                    f"{height:.1f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        # Calculate total hours of sleep per day, excluding the Awake phase
        sleep_columns = [
            "Deep Sleep",
            "Light Sleep",
            "REM Sleep",
        ]  # Only these phases contribute to sleep
        df["Total Sleep"] = df[sleep_columns].sum(axis=1)
        average_sleep = df["Total Sleep"].mean()
        plt.axhline(
            y=average_sleep,
            color="r",
            linestyle="--",
            label=f"Average Sleep ({average_sleep:.2f} hrs)",
        )

        plt.legend(title="Sleep Phases and Averages")
        plt.tight_layout()
        plt.show()

    def plot_sleep_habit_for_week(self):
        """
        y - hours when the patient goes to sleep and when he/she wakes up
        x - day for the week
        """
        # Extracting bedtime start and end times
        sleep_times = {"Day": [], "Sleep Start": [], "Wake Time": []}

        for entry in self.data:
            sleep_start = pd.to_datetime(entry["bedtime_start"])
            wake_time = pd.to_datetime(entry["bedtime_end"])
            fixed_date = datetime(2000, 1, 1)

            if sleep_start.date() != wake_time.date():
                # Sleep start to midnight
                sleep_times["Day"].append(sleep_start.date())
                sleep_times["Sleep Start"].append(
                    fixed_date
                    + timedelta(hours=sleep_start.hour, minutes=sleep_start.minute)
                )
                sleep_times["Wake Time"].append(
                    fixed_date + timedelta(hours=24)
                )  # Midnight as end time

                # Midnight to wake time
                sleep_times["Day"].append(wake_time.date())
                sleep_times["Sleep Start"].append(fixed_date)  # Start at midnight
                sleep_times["Wake Time"].append(
                    fixed_date
                    + timedelta(hours=wake_time.hour, minutes=wake_time.minute)
                )
            else:
                sleep_times["Day"].append(sleep_start.date())
                sleep_times["Sleep Start"].append(
                    fixed_date
                    + timedelta(hours=sleep_start.hour, minutes=sleep_start.minute)
                )
                sleep_times["Wake Time"].append(
                    fixed_date
                    + timedelta(hours=wake_time.hour, minutes=wake_time.minute)
                )

        # Creating DataFrame
        df_sleep = pd.DataFrame(sleep_times)
        df_sleep["Day"] = pd.Categorical(
            df_sleep["Day"], categories=pd.unique(df_sleep["Day"]), ordered=True
        )

        # Plot setup
        fig, ax = plt.subplots(figsize=(12, 6))

        start_time = datetime(2000, 1, 1, 0, 0)  # Start of the arbitrary day
        end_time = datetime(2000, 1, 1, 23, 59)  # End of the arbitrary day

        ax.yaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # Plot each sleep period as a vertical line on the graph
        for _, row in df_sleep.iterrows():
            start_time_num = mdates.date2num(row["Sleep Start"])
            end_time_num = mdates.date2num(row["Wake Time"])
            plt.vlines(
                x=row["Day"],
                ymin=start_time_num,
                ymax=end_time_num,
                colors="blue",
                lw=4,
            )

        # Customize the plot
        ax.set_ylabel("Time of Day (HH:MM)")
        ax.set_title("Daily Sleep Schedule Across the Week")
        fig.autofmt_xdate()  # Auto-format x-axis dates
        plt.ylim(
            [mdates.date2num(start_time), mdates.date2num(end_time)]
        )  # Set y-limits to cover one full day
        plt.tight_layout()
        plt.show()

    def plot_sleep_habit_for_week_polar(self):
        # Convert data into a DataFrame and convert times to datetime
        df = pd.DataFrame(self.data)
        df["bedtime_start"] = pd.to_datetime(df["bedtime_start"])
        df["bedtime_end"] = pd.to_datetime(df["bedtime_end"])
        df["day"] = pd.to_datetime(df["day"])

        df.sort_values("day", inplace=True)

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 10))
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")

        ax.set_facecolor("floralwhite")
        fig.set_facecolor("floralwhite")

        base_radius = 10  # Radius of the innermost circle
        width = 2

        alpha = 0.75
        color = "skyblue"
        edgecolor = "black"

        unique_days = df["day"].drop_duplicates().reset_index(drop=True)
        for index, day in enumerate(unique_days):
            day_data = df[df["day"] == day]
            radius = base_radius + index * width
            for _, row in day_data.iterrows():
                # Convert timezone-aware datetime to the correct number for plotting
                start_frac = (
                    row["bedtime_start"] - row["bedtime_start"].normalize()
                ).total_seconds() / 86400
                end_frac = (
                    row["bedtime_end"] - row["bedtime_end"].normalize()
                ).total_seconds() / 86400

                start_theta = start_frac * 2 * np.pi
                end_theta = end_frac * 2 * np.pi

                # If bedtime goes over midnight
                if start_theta > end_theta:
                    ax.barh(
                        radius,
                        2 * np.pi - start_theta,
                        left=start_theta,
                        height=width,
                        color=color,
                        alpha=alpha,
                        edgecolor=edgecolor,
                    )
                    ax.barh(
                        radius,
                        end_theta,
                        left=0,
                        height=width,
                        color=color,
                        alpha=alpha,
                        edgecolor=edgecolor,
                    )
                else:
                    ax.barh(
                        radius,
                        end_theta - start_theta,
                        left=start_theta,
                        height=width,
                        color=color,
                        alpha=alpha,
                        edgecolor=edgecolor,
                    )

                # Adding the day label inside the bar
                mid_theta = (
                    start_theta + (end_theta - start_theta) / 2
                    if start_theta <= end_theta
                    else np.pi
                )
                ax.text(
                    mid_theta,
                    radius + width / 2,
                    day,
                    color="black",
                    ha="center",
                    va="center",
                )

        ax.set_rticks([])  # Hide radial ticks
        ax.set_xticks(
            np.linspace(0, 2 * np.pi, 24, endpoint=False)
        )  # Set ticks every hour
        ax.set_xticklabels(
            [f"{(i % 24):02d}:00" for i in range(24)]
        )  # Label every hour

        # Title and labels
        plt.title(
            "Sleep Patterns Over Multiple Days",
            va="bottom",
            family="serif",
            fontsize=16,
        )
        plt.show()


class SleepDataOneDay:
    def __init__(self) -> None:
        pass
