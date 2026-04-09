# Plot and Email Criteria Reference

This document describes the exact criteria used to classify data, generate plots, and style the clinician email.

---

## Data Pre-processing

### MET 0.9 Bug Cleaning (activity.py)

During ingestion, consecutive runs of MET = 0.9 that are **>= 360 minutes (6 hours)** are replaced with NaN. Runs shorter than 360 minutes are left untouched.

- Threshold constant: `MET_09_CONSECUTIVE_THRESHOLD_MIN = 360`
- Affected days are logged and listed in the email footer as an informational note
- Does NOT trigger red highlighting or subject line warnings

### Day Boundaries

All analysis uses **12 PM to 12 PM** day windows (not midnight), since overnight sleep spans calendar dates. Steps and MET use the Oura default **4 AM to 4 AM** window.

---

## Sleep & Activity Gantt Chart (Top Plot)

Shows in-bed periods only. Each minute during in-bed is classified by `assign_state()` in this priority order:

| State | Condition | Color |
|-------|-----------|-------|
| Oura bug | `met_is_bug == True` (MET 0.9 bug was cleaned for this minute) | Orange (#ff9800), hatched |
| Not worn / battery dead | `activity_class == 0` OR `MET is NaN` OR `MET <= 0.1` | Gray (#aaaaaa), hatched |
| Deep sleep | `sleep_phase_label == "deep"` | Dark blue (#0b3d91) |
| Light sleep | `sleep_phase_label == "light"` | Medium blue (#3c82e0) |
| REM sleep | `sleep_phase_label == "REM"` | Light blue (#9ec5f2) |
| Awake | `sleep_phase_label == "awake"` | Light grey (#e6e6e6) |
| Unidentified | In bed but none of the above match | Soft yellow (#ffe17b) |

The last day (yesterday) is rendered at 40% opacity with "Excluded from analysis" label.

### Estimated Rest Underbar

A thin green bar drawn below each day's sleep bar showing periods of low MET activity.

| | Criterion |
|---|-----------|
| **Condition** | `MET > 0.1` AND `MET < 1.1` AND `MET is not NaN` |
| **Minimum duration** | Segments shorter than 10 minutes are filtered out to remove speckles |
| **Color** | Light green (#90EE90), alpha 0.85 (0.4 for yesterday) |
| **Position** | Thin bar (height 0.25) below the main sleep bar |
| **Purpose** | Shows rest periods inferred from MET so clinicians can cross-reference with sleep data |

- MET <= 0.1 is excluded (indicates non-wear, not rest)
- NaN MET is excluded (no data or cleaned MET 0.9 bug)
- Threshold constant: `MET_REST_THRESHOLD = 1.1`
- Duration constant: `MET_REST_MIN_DURATION_MIN = 10`

---

## MET Intensity Gantt Chart (Bottom Plot)

Shows MET values for all time (not just in-bed). Each minute is bucketed by `bucketize_met()`:

| Bucket | Condition | Color |
|--------|-----------|-------|
| Oura bug | `met_is_bug == True` | Orange (#ff9800), hatched |
| Non worn / battery dead | `MET is NaN` OR `MET <= 0.1` | Gray (#aaaaaa), hatched |
| 0.1 - 0.5 | `0.1 < MET < 0.5` | Very light blue (#deebf7) |
| 0.5 - 1.0 | `0.5 <= MET < 1.0` | Light blue (#9ecae1) |
| 1.0 - 2.0 | `1.0 <= MET < 2.0` | Medium blue (#6baed6) |
| 2.0 - 3.0 | `2.0 <= MET < 3.0` | Standard blue (#4292c6) |
| 3.0 - 4.5 | `3.0 <= MET < 4.5` | Deep blue (#2171b5) |
| > 4.5 | `MET >= 4.5` | Darkest blue (#084594) |

---

## Email Table: Red Highlighting

A cell is highlighted red (`#ff5252`) if any of its associated warning flags are True.

| Column | Red when |
|--------|----------|
| Sleep (12pm-12pm, Day-2 to Yesterday) | `lastday_sleep is NaN` OR `lastday_sleep < 6 hours` OR `sleep_variation > ±25%` |
| Daily Sleep Score | `lastday_sleep_score is NaN` |
| Average Sleep (h) | `average_sleep is NaN` |
| Average Sleep Score | `average_sleep_score is NaN` |
| Day-2 Steps | *(no warning flags currently active)* |
| Average Steps | *(no warning flags currently active)* |
| Day-2 Average MET | `lastday_met is NaN` OR `met_variation > ±25%` |
| Average MET | `average_met is NaN` |

### Variation formula

`sleep_variation` triggers when `lastday_sleep < 0.75 * average_sleep` OR `lastday_sleep > 1.25 * average_sleep` (and both values are non-NaN).

`met_variation` uses the same ±25% formula applied to MET.

### Non-wear warning

`lastday_non_wear_time_over_8` triggers when `non_wear_time > 28800 seconds (8 hours)`. This is used for Qualtrics survey triggers but does not currently red-highlight any email cell.

---

## Email Table: Inactive Patient Styling

Configured via `inactive_patients` dict in config.json (optional, defaults to `{}`). Keys are patient IDs, values are the date they were marked inactive (e.g., `{"TRBD003": "2026-04-06"}`).

| Condition | Styling |
|-----------|---------|
| Inactive + no lastday data (sleep and MET both NaN) | Gray text (#999999), no red highlighting |
| Inactive + has lastday data AFTER inactive date (sleep or MET is non-NaN, and lastday > inactive_since) | Blue background (#64B5F6), white text |
| Active | Normal behavior (red on warning) |

- Inactive patients are labeled "(inactive)" in the Patient column
- Inactive patients are excluded from subject line warnings unless unexpected data appeared
- Qualtrics survey triggers are skipped for inactive patients

---

## Email Subject Line

| Condition | Format |
|-----------|--------|
| No active patients have warnings | `{study_name} [All Clear] for Patients: {patient_list}` |
| Some active patients have warnings | `{study_name} [Warning: {flagged_patients} need review]` |
| Inactive patient has unexpected data | Added as `{patient} (inactive-data)` to flagged list |

A patient is "flagged" if ANY of their warning flags are True, OR if a Qualtrics survey completed yesterday has warnings.

---

## Email Footer Notes

- **MET 0.9 Bug** (shown when any patient is affected): "Some days had extended periods of MET=0.9 (Oura autofill when ring is not worn). These values were excluded from averages."
- **Survey score thresholds** (shown when survey patients are included): ISS activation/well-being thresholds for mood states, PHQ-8 > 10, ASRM > 6.
