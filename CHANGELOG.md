# Changelog

## 2026-04-08 — MET Bug Detection, Rest Overlay, Inactive Patients

**Author:** Yewen Zhou + Claude

### Problem

1. **MET 0.9 Bug:** Oura sometimes auto-fills MET values with 0.9 when the ring is not worn, instead of reporting NaN or proper no-wear indicators. This polluted MET averages and hid non-wear periods. Data analysis found 91 out of 1392 daily activity files affected, with some showing 24 hours of continuous 0.9.

2. **No-sleep ambiguity:** When sleep data is NaN for a day, clinicians couldn't tell if the patient didn't wear the ring, didn't sync, or simply didn't sleep. There was no visual context from other signals.

3. **Inactive patient alert fatigue:** Patients who stop wearing rings generate constant red warnings in the email, creating noise that makes it harder to spot real issues.

### Goal

- Detect and clean the MET 0.9 bug so it doesn't corrupt downstream statistics
- Give clinicians a visual "estimated rest" signal from MET data to cross-reference with sleep
- Allow marking patients as inactive to suppress unnecessary warnings while still monitoring them
- Design (but not implement) webhook integration points for a coworker's sync-tracking work

### Plan

- **Phase 1a:** Detect consecutive MET=0.9 runs >= 6 hours at ingestion time in `activity.py`, replace with NaN. The 6-hour threshold was empirically validated: the longest legitimate consecutive 0.9 run across 1305 normal days was ~5.9 hours (99th percentile: 4.2 hours), while all 91 bug days had runs >= 6 hours.

- **Phase 1b:** Show MET-inferred rest (MET < 1.2) as a thin green bar below each day's sleep bar on the Gantt chart. This gives clinicians a second signal to compare against sleep — visible especially on days with missing sleep data.

- **Phase 2:** Add `inactive_patients` config list. Inactive patients are still processed but their email rows are grayed out (no red highlighting). If data unexpectedly appears for an inactive patient, the row highlights blue. Inactive patients are excluded from the email subject line warnings and Qualtrics survey triggers.

- **Phase 3 (design only):** Documented integration points for future webhook-based sync tracking: `trbdv0/webhook.py` module, "Last Sync" email column, sync timeline plot, and `enable_webhook_features` config flag.

### What Changed

**`trbdv0/constants.py`**
- Added `MET_09_CONSECUTIVE_THRESHOLD_MIN`, `MET_09_BUG_DETECTED`, `MET_09_BUG_DATES`
- Added `MET_REST_THRESHOLD`
- Added `IS_INACTIVE`, `HAS_DATA_WHILE_INACTIVE`

**`trbdv0/activity.py`**
- Added `_detect_and_clean_met_09_bug()` method: walks MET items, finds consecutive 0.9 runs >= threshold, replaces with NaN
- Integrated into `ingest()` after existing MET trim logic
- Tracks affected days in `self.met_09_bug_days`

**`trbdv0/master.py`**
- `get_summary_stats()`: propagates `MET_09_BUG_DATES` and `MET_09_BUG_DETECTED`
- `generate_warning_flags()`: added `MET_09_BUG_DETECTED` flag
- `plot_integrated_sleep_activity_schedule()`: added estimated rest underbar (thin green bars, height=0.15, y-offset=-0.3) showing periods where MET < 1.2, with legend entry
- No changes needed for MET Gantt — NaN values already map to gray "non_worn/battery_dead"

**`trbdv0/main.py`**
- Loops over `active_patients + inactive_patients`
- Sets `IS_INACTIVE` and `HAS_DATA_WHILE_INACTIVE` on patient summary stats
- Skips Qualtrics survey triggers for inactive patients

**`trbdv0/send_email.py`**
- `style()`: accepts `is_inactive` and `has_data_inactive` kwargs
  - Inactive + no data → gray text (`#999999`)
  - Inactive + has data → blue background (`#64B5F6`)
  - Active → unchanged (red on warning)
- MET columns now trigger on `MET_09_BUG_DETECTED`
- Added MET 0.9 bug footer note when any patient is affected
- `generate_subject_line()`: excludes inactive patients unless unexpected data appeared

**`config/config_test_trbd.json`**
- Added `inactive_patients` list (optional key, defaults to `[]` for backward compatibility)
- Moved TRBD003 from `active_patients` to `inactive_patients`
