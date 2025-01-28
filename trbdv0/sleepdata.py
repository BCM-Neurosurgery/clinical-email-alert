import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
from matplotlib.cm import get_cmap
import os
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from trbdv0.utils import (
    get_todays_date,
    get_yesterdays_date,
    PHASE_MAPPING,
)

# TODO: how to define a day?


class SleepData:
    def __init__(self, patient, data: list) -> None:
        """Init class

        Args:
            patient (str): e.g. Percept010
            data (list): list of dicts of sleep info
        """
        # self.data looks like [{}, {}, ..., {}]
        self.data = data
        # self.sleep_data_one_day -> SleepDataOneDay object
        self.sleep_data_one_day = {}
        self.num_past_days = len(self.get_available_dates())
        self.patient = patient
        self.summary_stats = self.get_summary_stats()

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

    def get_summary_stats(self) -> dict:
        """Get summary stats of sleep data

        Returns:
            dict: {
                "patient": "Percept010",
                "date_range": ("2023-09-01", "2023-09-14"), # available range in sleep data
                "today": "2023-09-14", # might not be in the data
                "yesterday": "2023-09-13",
                "yesterday_sleep": 7.5,
                "average_sleep": 7.5, # average sleep in sleep data
                "sleep_df": pd.DataFrame,
            }
        """
        # Collect counters by day
        sleep_counts = {}
        for entry in self.data:
            day = entry["day"]
            if day not in sleep_counts:
                sleep_counts[day] = Counter()
            sleep_counts[day] += Counter(entry["sleep_phase_5_min"])
            sleep_counts[day]["non_wear_time"] = entry.get("non_wear_time", 0) / 3600
            sleep_counts[day]["steps"] = entry.get("steps", 0)

        # Convert 5-min increments to hours
        for day, counter in sleep_counts.items():
            for phase in list(counter.keys()):
                if phase not in ["non_wear_time", "steps"]:
                    counter[phase] = (counter[phase] * 5) / 60

        # Flatten data into a DataFrame
        all_keys = ["1", "2", "3", "4", "non_wear_time", "steps"]
        flattened_data = {
            d: {k: sleep_counts[d].get(k, np.nan) for k in all_keys}
            for d in sleep_counts
        }
        df = pd.DataFrame.from_dict(flattened_data, orient="index").sort_index()
        df.columns = [PHASE_MAPPING.get(col, f"Phase {col}") for col in df.columns]

        # Reorder columns and calculate total sleep
        column_order = [
            "Deep Sleep",
            "Light Sleep",
            "REM Sleep",
            "Awake",
            "Non-Wear Time",
            "Step Count",
        ]
        df = df.reindex(columns=column_order).fillna(0)
        df["Total Sleep"] = df[["Deep Sleep", "Light Sleep", "REM Sleep"]].sum(axis=1)

        # Get yesterday's sleep and average
        today_date = get_todays_date()
        yesterday_date = get_yesterdays_date()
        yesterday_sleep = (
            df.loc[yesterday_date, "Total Sleep"]
            if yesterday_date in df.index
            else np.nan
        )
        yesterday_step = (
            df.loc[yesterday_date, "Step Count"]
            if yesterday_date in df.index
            else np.nan
        )

        return {
            "patient": self.patient,
            "date_range": (df.index.min(), df.index.max()),
            "today": today_date,
            "yesterday": yesterday_date,
            "yesterday_sleep": yesterday_sleep,
            "yesterday_steps": yesterday_step,
            "average_sleep": df["Total Sleep"].mean() if not df.empty else np.nan,
            "average_steps": df["Step Count"].mean() if not df.empty else np.nan,
            "sleep_df": df,
        }

    def plot_combined_sleep_plots(self, out_dir: str):
        """
        Plot sleep distribution and sleep habit polar plot side by side.
        """
        df = self.summary_stats["sleep_df"].drop(columns=["Step Count", "Total Sleep"])
        # Create a figure with two subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(26, 10))

        # Plot sleep distribution on the first subplot
        ax1 = axes[0]
        color_palette = list(get_cmap("Set2").colors[:4]) + ["lightgray"]
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

        average_sleep = self.summary_stats["average_sleep"]
        ax1.axhline(
            y=average_sleep,
            color="r",
            linestyle="--",
            label=f"Average Sleep ({average_sleep:.2f} hrs)",
        )

        ax1.legend(title="Sleep Phases and Averages")

        # Plot spiral chart
        dataframe = pd.DataFrame(self.data)
        dataframe["day"] = pd.to_datetime(dataframe["day"])
        dataframe["bedtime_start"] = pd.to_datetime(
            dataframe["bedtime_start"].str.slice(0, 19)
        )
        dataframe["bedtime_end"] = pd.to_datetime(
            dataframe["bedtime_end"].str.slice(0, 19)
        )
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"].str.slice(0, 19))

        # Find the minimum date from bedtime_start
        dataframe = dataframe.dropna()
        min_day = dataframe["bedtime_start"].dt.date.min()

        colors = list(mcolors.TABLEAU_COLORS.values())
        day_colors = {
            day: colors[i % len(colors)]
            for i, day in enumerate(dataframe["day"].unique())
        }

        ax2 = fig.add_subplot(122, projection="polar")
        ax2.set_theta_direction(-1)
        ax2.set_theta_offset(np.pi / 2.0)
        ax2.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
        ax2.set_xticklabels([f"{i}:00" for i in range(24)])
        ax2.set_yticklabels([])

        ax2.grid(False)
        ax2.xaxis.grid(True, color="black", linestyle="-", linewidth=0.5, alpha=0.8)

        def plot_period(
            start_time,
            end_time,
            start_offset,
            color,
            linestyle="-",
            lw=10,
            alpha=1,
            edgecolor=None,
        ):
            start_angle = (start_time.hour * 60 + start_time.minute) / 1440 * 2 * np.pi
            end_angle = (end_time.hour * 60 + end_time.minute) / 1440 * 2 * np.pi

            if end_angle < start_angle:
                end_angle += 2 * np.pi

            start_radius = start_offset + start_angle / (2 * np.pi)
            end_radius = start_offset + end_angle / (2 * np.pi)

            angles = np.linspace(
                start_angle + 2 * np.pi * start_offset,
                end_angle + 2 * np.pi * start_offset,
                100,
            )
            radii = np.linspace(start_radius, end_radius, 100)

            if edgecolor:
                ax2.plot(
                    angles,
                    radii,
                    color=edgecolor,
                    linewidth=lw,
                    linestyle=linestyle,
                    alpha=alpha,
                )
            ax2.plot(
                angles,
                radii,
                color=color,
                linewidth=lw,
                linestyle=linestyle,
                alpha=alpha,
            )

        dataframe = dataframe.sort_values(by="bedtime_start")
        awake_periods = []
        non_worn_periods = []

        for index in range(len(dataframe)):
            row = dataframe.iloc[index]
            bedtime_start = row["bedtime_start"]
            bedtime_end = row["bedtime_end"]
            day_color = day_colors[row["day"]]
            class_5_min = row["class_5_min"]
            timestamp = row["timestamp"]

            start_offset = (bedtime_start.date() - min_day).days

            # Plot sleep period
            plot_period(bedtime_start, bedtime_end, start_offset, day_color)

            # Collect awake periods if not the last row
            if index < len(dataframe) - 1:
                next_row = dataframe.iloc[index + 1]
                next_bedtime_start = next_row["bedtime_start"]
                awake_start = bedtime_end
                awake_end = next_bedtime_start
                awake_start_offset = (awake_start.date() - min_day).days

                if awake_start.date() == awake_end.date():
                    awake_periods.append((awake_start, awake_end, awake_start_offset))
                else:
                    # Awake period spans multiple days
                    midnight = awake_start.replace(hour=23, minute=59, second=59)
                    awake_periods.append((awake_start, midnight, awake_start_offset))
                    next_start_offset = (awake_end.date() - min_day).days
                    next_midnight = awake_end.replace(hour=0, minute=0, second=0)
                    awake_periods.append((next_midnight, awake_end, next_start_offset))

            # Collect non-worn periods as continuous segments
            start_time = timestamp
            current_date = start_time.date()
            non_worn_offset = (current_date - min_day).days
            non_worn_start = None
            for i, status in enumerate(class_5_min):
                end_time = start_time + pd.Timedelta(minutes=5)
                if end_time.date() != current_date:
                    non_worn_offset += 1
                    current_date = end_time.date()
                if status == "0":
                    if non_worn_start is None:
                        non_worn_start = start_time
                else:
                    if non_worn_start is not None:
                        non_worn_periods.append(
                            (non_worn_start, start_time, non_worn_offset)
                        )
                        non_worn_start = None
                start_time = end_time
            # Add the last segment if it ends in a non-worn period
            if non_worn_start is not None:
                non_worn_periods.append((non_worn_start, start_time, non_worn_offset))

        # Plot awake periods first
        for start_time, end_time, start_offset in awake_periods:
            plot_period(
                start_time,
                end_time,
                start_offset,
                "white",
                linestyle="-",
                edgecolor="black",
            )
        # Plot non-worn periods last to ensure they are on top
        for start_time, end_time, start_offset in non_worn_periods:
            plot_period(
                start_time, end_time, start_offset, "lightgray", edgecolor="black"
            )

        # Custom legend
        custom_lines = [
            Line2D(
                [0],
                [0],
                color="white",
                lw=4,
                linestyle="--",
            ),
            Line2D([0], [0], color="lightgray", lw=4),
        ] + [
            Line2D([0], [0], color=day_colors[day], lw=4, linestyle="-")
            for day in dataframe["day"].unique()
        ]
        custom_labels = ["Awake Period", "Non-worn Period"] + [
            f"{day.date()}" for day in dataframe["day"].unique()
        ]
        ax2.legend(custom_lines, custom_labels, loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        # Save the combined figure
        plt.savefig(
            os.path.join(
                out_dir,
                f"{self.patient}.png",
            )
        )
