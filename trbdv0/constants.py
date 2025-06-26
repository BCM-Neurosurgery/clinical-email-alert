# keys for master summary
PATIENT = "patient"
STUDY_NAME = "study_name"
TODAYS_DATE = "todays_date"
LASTDAY_DATE = "lastday_date"

LASTDAY_SLEEP_HOURS = "lastday_sleep_hours"
AVERAGE_SLEEP_HOURS = "average_sleep_hours"

LASTDAY_NON_WEAR_TIME_S = "lastday_non_wear_time_s"

LASTDAY_STEPS = "lastday_steps"
AVERAGE_STEPS = "average_steps"

LASTDAY_MET = "lastday_met"
AVERAGE_MET = "average_met"

MISSING_SLEEP_DATES = "missing_sleep_dates"
NUMBER_OF_NANSLEEP_DAYS = "number_of_nansleep_days"
NUMBER_OF_DAYS = "number_of_days"

# keys for warning flags
LASTDAY_SLEEP_NAN = "lastday_sleep_nan"
AVERAGE_SLEEP_NAN = "average_sleep_nan"
HAS_NAN_SLEEP_DAYS = "has_nansleep_days"
LASTDAY_SLEEP_LESS_THAN_6 = "lastday_sleep_less_than_6"
SLEEP_VARIATION = "sleep_variation"

LASTDAY_STEPS_NAN = "lastday_steps_nan"
AVERAGE_STEPS_NAN = "average_steps_nan"
STEPS_VARIATION = "steps_variation"

LASTDAY_MET_NAN = "lastday_met_nan"
AVERAGE_MET_NAN = "average_met_nan"
MET_VARIATION = "met_variation"

LASTDAY_NON_WEAR_TIME_OVER_8 = "lastday_non_wear_time_over_8"

# keys for email body columns
PT_COLUMN = "Patient"
MISSING_LASTDAY_SLEEP_COLUMN = "Missing Last Day Sleep"

LASTDAY_SLEEP_COLUMN = "Sleep (12pm-12pm, Day-2 to Yesterday)"
AVERAGE_SLEEP_COLUMN = "Average Sleep (h)"

LASTDAY_STEPS_COLUMN = "Yesterday Steps"
AVERAGE_STEPS_COLUMN = "Average Steps"

LASTDAY_MET_COLUMN = "Yesterday Average MET"
AVERAGE_MET_COLUMN = "Average MET"

# a mapping of survey IDs to processor classes
SURVEY_CLASSES = {
    "PHQ-8": "trbdv0.survey_processor.PHQ8Processor",
    "ASRM": "trbdv0.survey_processor.ASRMProcessor",
    "ISS": "trbdv0.survey_processor.ISSProcessor",
}

# map each survey → which SC keys to highlight when any warnings fire
HIGHLIGHT_KEYS = {
    "ISS": ["SC1", "SC2"],
    "PHQ-8": ["SC0"],
    "ASRM": ["SC0"],
}

# currently only include TRBD patients for
# email qualtric survey
ALLOWED_SURVEY_PATIENTS = {"TRBD001", "TRBD002"}

# LFP related constants
LFP_CONSTANTS = {
    "009": {
        "directory": "/mnt/datalake/data/PerceptOCD-48392/009/LFP/",
        "dbs_date": "2020-08-11",
        "response_status": 1,
        "response_date": 241
    },
    "DBSOCD001": {
        "directory": "/mnt/datalake/data/DBSPsych-56119/DBSOCD001/LFP/",
        "dbs_date": "2025-01-28",
        "response_status": 0
    },
    "DBSOCD002": {
        "directory": "/mnt/datalake/data/DBSPsych-56119/DBSOCD002/LFP/",
        "dbs_date": "2025-04-10",
        "response_status": 0
    },
    "DBSOCD004": {
        "directory": "/mnt/datalake/data/DBSPsych-56119/DBSOCD004/LFP/",
        "dbs_date": "2025-04-10",
        "response_status": 0
    },
    "TRBD001": {
        "directory": "/mnt/datalake/data/TRBD-53761/TRBD001/LFP/",
        "dbs_date": "2025-06-04",
        "response_status": 0
    },
    "TRBD002": {
        "directory": "/mnt/datalake/data/TRBD-53761/TRBD002/LFP/",
        "dbs_date": "2025-07-09",
        "response_status": 0
    }
}