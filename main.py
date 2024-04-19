from sleepdata import SleepData


if __name__ == "__main__":
    path = "Percept009_Sleep.json"
    data = SleepData(path)
    day = "2023-09-14"
    # print(data.get_available_days())
    # print(data.plot_sleep_interval_on_day("2023-09-14"))
    print(data.plot_sleep_phase_5_min(day))
    print(data.get_summary_stat_for_day(day))
