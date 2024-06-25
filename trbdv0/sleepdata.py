import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import matplotlib.dates as mdates
import trbdv0.utils as utils
import pandas as pd
from collections import Counter
import numpy as np
from matplotlib.cm import get_cmap
import os

# TODO: how to define a day?


class SleepData:
    def __init__(self, data: list) -> None:
        """Init class

        Args:
            data (list): list of dicts of sleep info
        """
        # self.data looks like [{}, {}, ..., {}]
        self.data = data
        # self.sleep_data_one_day -> SleepDataOneDay object
        self.sleep_data_one_day = {}
        self.num_past_days = len(self.data)

    def get_num_past_days(self):
        return self.num_past_days

    def get_available_dates(self) -> set:
        """Return all recorded dates

        Returns:
            set: set("2023-07-15", "2023-07-16")
        """
        res = set()
        for sleep_chunk in self.data:
            res.add(sleep_chunk["day"])
        return res

    def get_sleep_data_on_date(self, date: str) -> list:
        """Return a list of sleep data on that day

        Args:
            day (str): "2023-09-14"

        Returns:
            list: [{}, {}, ...]
        """
        res = []
        for sleep_chunk in self.data:
            if sleep_chunk["day"] == date:
                res.append(sleep_chunk)
        return res

    def plot_sleep_interval_on_date(self, ax, date: str) -> None:
        """Plot sleep intervals on date"""
        sleep_data_on_day = self.get_sleep_data_on_date(date)

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
        ax.set_title(f"Sleep Intervals for {date}")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        # by default it plots in UTC time
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        # Improve layout and show grid
        plt.grid(True)

    def get_summary_stat_for_date(self, day: str) -> pd.DataFrame:
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
        sleep_data_on_day = self.get_sleep_data_on_date(day)
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

    def get_summary_plot_for_date(self, date: str, out_dir: str):
        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Call the first function and pass its ax to the first subplot
        self.plot_sleep_interval_on_date(axes[0], date)
        self.plot_sleep_phase_5_min(axes[1], date)
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plot_file = os.path.join(out_dir, f"sleep_interval_{date}.png")
        plt.savefig(plot_file)

        # save the 2nd one independently
        self.plot_sleep_distribution_for_date(date, out_dir)

    def plot_sleep_phase_5_min(self, ax, date):
        sleep_data_on_day = self.get_sleep_data_on_date(date)

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

    def plot_sleep_distribution_for_date(self, date, out_dir: str):
        """
        Plot the distribution of sleep phase per day
        """
        # Initialize a dictionary to hold percentage data for each sample
        percentage_data = []

        sleep_data_on_day = self.get_sleep_data_on_date(date)

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
        color_palette = get_cmap("Set2").colors[:4]
        for i, phase in enumerate(phases):
            bar_positions = x_positions + i * bar_width
            bars = ax.bar(
                bar_positions,
                plot_data[phase],
                width=bar_width,
                color=color_palette[i],
                alpha=0.7,
                edgecolor="black",
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
        plot_file = os.path.join(out_dir, f"sleep_distribution_{date}.png")
        plt.savefig(plot_file)
        plt.close()

    def plot_sleep_distribution(self, out_dir: str, past_days: int):
        """
        Plot sleep distribution for the past past_days
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

        all_keys = ["1", "2", "3", "4"]
        flattened_data = {
            date: {key: sleep_data[date].get(key, np.nan) for key in all_keys}
            for date in sleep_data
        }
        df = pd.DataFrame.from_dict(flattened_data, orient="index")
        df = df.sort_index(ascending=True)
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
        color_palette = get_cmap("Set2").colors[:4]
        ax = df.plot(
            kind="bar",
            stacked=True,
            color=color_palette,
            alpha=0.7,
            edgecolor="black",
            figsize=(10, 7),
        )
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
        average_sleep = df.dropna()["Total Sleep"].mean()
        plt.axhline(
            y=average_sleep,
            color="r",
            linestyle="--",
            label=f"Average Sleep ({average_sleep:.2f} hrs)",
        )

        plt.legend(title="Sleep Phases and Averages")
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"sleep_distribution_past_{past_days}_days.png")
        )

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

    def plot_sleep_habit_polar(self, out_dir: str):
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
        edgecolor = "black"
        color_palette = get_cmap("Set2").colors[:7]

        unique_days = df["day"].drop_duplicates().reset_index(drop=True)
        for index, day in enumerate(unique_days):
            color = color_palette[index % 7]
            day_data = df[df["day"] == day]
            radius = base_radius + index * width
            for _, row in day_data.iterrows():
                if pd.isna(row["bedtime_start"]) and pd.isna(row["bedtime_end"]):
                    ax.barh(
                        radius,
                        0,
                        left=0,
                        height=width,
                        color=color,
                        alpha=alpha,
                        edgecolor=edgecolor,
                    )
                    continue

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

        # Create the legend
        day_labels = [day.strftime("%a %Y-%m-%d") for day in unique_days]
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_palette[i % 7], alpha=alpha)
            for i in range(len(unique_days))
        ]
        plt.legend(handles, day_labels, loc="upper left", bbox_to_anchor=(1, 1))

        ax.set_rticks([])  # Hide radial ticks
        ax.set_xticks(
            np.linspace(0, 2 * np.pi, 24, endpoint=False)
        )  # Set ticks every hour
        ax.set_xticklabels(
            [f"{(i % 24):02d}:00" for i in range(24)]
        )  # Label every hour

        # Title and labels
        plt.title(
            f"Sleep Patterns Over Past {self.get_num_past_days()} Days",
            va="bottom",
            family="serif",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir, f"sleep_habit_past_{self.get_num_past_days()}_days.png"
            )
        )

    def plot_combined_sleep_plots(self, out_dir: str):
        """
        Plot sleep distribution and sleep habit polar plot side by side.
        """
        sleep_data = {}

        for entry in self.data:
            day = entry["day"]
            sleep_phases = entry["sleep_phase_5_min"]
            phase_counts = Counter(sleep_phases)  # Count occurrences of each phase

            if day not in sleep_data:
                sleep_data[day] = Counter()

            sleep_data[day] += phase_counts

            # Add non_wear_time in hours
            non_wear_time_seconds = entry.get("non_wear_time", 0)
            sleep_data[day]["non_wear_time"] = (
                non_wear_time_seconds / 3600
            )  # Convert to hours

        # Convert counts to hours (assuming each phase count is in 5 minutes increments)
        for day in sleep_data:
            for phase in sleep_data[day]:
                if (
                    phase != "non_wear_time"
                ):  # Skip non_wear_time as it's already in hours
                    sleep_data[day][phase] = (
                        sleep_data[day][phase] * 5
                    ) / 60  # Convert to hours

        all_keys = ["1", "2", "3", "4", "non_wear_time"]
        flattened_data = {
            date: {key: sleep_data[date].get(key, np.nan) for key in all_keys}
            for date in sleep_data
        }
        df = pd.DataFrame.from_dict(flattened_data, orient="index")
        df = df.sort_index(ascending=True)
        # Rename columns based on sleep phase descriptions
        phase_mapping = {
            "1": "Deep Sleep",
            "2": "Light Sleep",
            "3": "REM Sleep",
            "4": "Awake",
            "non_wear_time": "Non-Wear Time",
        }
        df.columns = [phase_mapping.get(col, f"Phase {col}") for col in df.columns]

        # Reorder so that awake and non-wear time are at the top
        column_order = [
            "Deep Sleep",
            "Light Sleep",
            "REM Sleep",
            "Awake",
            "Non-Wear Time",
        ]
        df = df[column_order]

        # Create a figure with two subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(26, 10))

        # Plot sleep distribution on the first subplot
        ax1 = axes[0]
        color_palette = get_cmap("Set2").colors[:5]
        ax1 = df.plot(
            kind="bar",
            stacked=True,
            color=color_palette,
            alpha=0.7,
            edgecolor="black",
            ax=ax1,
        )
        ax1.set_title("Distribution of Sleep Phases per Day")
        ax1.set_ylabel("Hours")
        ax1.set_xlabel("Day")
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df.index, rotation=45)

        # Adding labels to each bar
        for p in ax1.patches:
            width, height = p.get_width(), p.get_height()
            if (
                height > 0
            ):  # Only label the bar if the height is significant to avoid clutter
                x, y = p.get_xy()
                ax1.text(
                    x + width / 2,
                    y + height / 2,
                    f"{height:.1f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        # Calculate total hours of sleep per day, excluding the Awake and Non-Wear Time phases
        sleep_columns = [
            "Deep Sleep",
            "Light Sleep",
            "REM Sleep",
        ]  # Only these phases contribute to sleep
        df["Total Sleep"] = df[sleep_columns].sum(axis=1)
        average_sleep = df.dropna()["Total Sleep"].mean()
        ax1.axhline(
            y=average_sleep,
            color="r",
            linestyle="--",
            label=f"Average Sleep ({average_sleep:.2f} hrs)",
        )

        ax1.legend(title="Sleep Phases and Averages")

        # Plot sleep habit polar plot on the second subplot
        df2 = pd.DataFrame(self.data)
        df2["bedtime_start"] = pd.to_datetime(df2["bedtime_start"])
        df2["bedtime_end"] = pd.to_datetime(df2["bedtime_end"])
        df2["day"] = pd.to_datetime(df2["day"])

        df2.sort_values("day", inplace=True)

        ax2 = axes[1]
        ax2 = fig.add_subplot(122, projection="polar")
        ax2.set_theta_direction(-1)
        ax2.set_theta_zero_location("N")

        ax2.set_facecolor("floralwhite")
        fig.set_facecolor("floralwhite")

        base_radius = 10  # Radius of the innermost circle
        width = 2

        alpha = 0.75
        edgecolor = "black"
        color_palette = get_cmap("Set2").colors[:7]

        unique_days = df2["day"].drop_duplicates().reset_index(drop=True)
        for index, day in enumerate(unique_days):
            color = color_palette[index % 7]
            day_data = df2[df2["day"] == day]
            radius = base_radius + index * width
            for _, row in day_data.iterrows():
                if pd.isna(row["bedtime_start"]) and pd.isna(row["bedtime_end"]):
                    ax2.barh(
                        radius,
                        0,
                        left=0,
                        height=width,
                        color=color,
                        alpha=alpha,
                        edgecolor=edgecolor,
                    )
                    continue

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
                    ax2.barh(
                        radius,
                        2 * np.pi - start_theta,
                        left=start_theta,
                        height=width,
                        color=color,
                        alpha=alpha,
                        edgecolor=edgecolor,
                    )
                    ax2.barh(
                        radius,
                        end_theta,
                        left=0,
                        height=width,
                        color=color,
                        alpha=alpha,
                        edgecolor=edgecolor,
                    )
                else:
                    ax2.barh(
                        radius,
                        end_theta - start_theta,
                        left=start_theta,
                        height=width,
                        color=color,
                        alpha=alpha,
                        edgecolor=edgecolor,
                    )

                class_5_mins = row.get("class_5_min", "")
                timestamp = row.get("timestamp")
                if class_5_mins and timestamp:
                    start_time = pd.to_datetime(timestamp).time()
                    start_time_seconds = (
                        start_time.hour * 3600
                        + start_time.minute * 60
                        + start_time.second
                    )
                    start_offset = start_time_seconds / 86400 * 2 * np.pi

                    # Find continuous non-wear periods
                    non_wear_periods = []
                    current_period = None
                    for i, c in enumerate(class_5_mins):
                        if c == "0":
                            if current_period is None:
                                current_period = [i]
                        else:
                            if current_period is not None:
                                current_period.append(i)
                                non_wear_periods.append(current_period)
                                current_period = None
                    if current_period is not None:
                        current_period.append(len(class_5_mins))
                        non_wear_periods.append(current_period)

                    for start, end in non_wear_periods:
                        start_theta = (start / 288) * 2 * np.pi + start_offset
                        end_theta = (end / 288) * 2 * np.pi + start_offset
                        ax2.barh(
                            radius,
                            end_theta - start_theta,
                            left=start_theta,
                            height=width / 2,  # Half the width for non-wear time
                            color="white",
                            alpha=alpha,
                            edgecolor=edgecolor,
                        )
        # Create the legend
        day_labels = [day.strftime("%a %Y-%m-%d") for day in unique_days]
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_palette[i % 7], alpha=alpha)
            for i in range(len(unique_days))
        ]
        non_wear_handle = plt.Rectangle((0, 0), 1, 1, color="white", alpha=alpha)
        handles.append(non_wear_handle)
        day_labels.append("Non-Wear Time")
        plt.legend(handles, day_labels, loc="upper left", bbox_to_anchor=(1, 1))

        ax2.set_rticks([])  # Hide radial ticks
        ax2.set_xticks(
            np.linspace(0, 2 * np.pi, 24, endpoint=False)
        )  # Set ticks every hour
        ax2.set_xticklabels(
            [f"{(i % 24):02d}:00" for i in range(24)]
        )  # Label every hour

        # Adjust subplot spacing
        plt.subplots_adjust(wspace=0.4)

        # Title and labels
        ax2.set_title(
            f"Sleep Patterns Over Past {self.get_num_past_days()} Days",
            va="bottom",
            family="serif",
            fontsize=16,
        )
        plt.tight_layout()

        # Save the combined figure
        plt.savefig(
            os.path.join(
                out_dir,
                f"combined_sleep_plots_past_{self.get_num_past_days()}_days.png",
            )
        )

        return {
            "average_sleep": average_sleep,
        }
