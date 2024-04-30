from sleepdata import SleepData


if __name__ == "__main__":
    path = "Percept009_Sleep.json"
    data = SleepData(path)
    day = "2023-09-14"
    """
    plot sleep schedule for a certain date
    """
    # data.get_summary_stat_for_day(day)
    data.get_summary_plot_for_day(day)

    """
    plot sleep distribution for the past week
    y - hours int
    x - day
    """
    data.plot_sleep_distribution_for_week()

    """
    plot when the patient goes to sleep every day
    y - clock datetime
    x - day
    """
    # data.plot_sleep_habit_for_week()
    data.plot_sleep_habit_for_week_polar()
