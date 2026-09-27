# MetroPT-3 Local Dataset Analysis & Audit

## Summary

This document records the empirical verification and metadata audit of the local dataset for the MetroGuard project. All statistics are verified directly from `data/raw/MetroPT3(AirCompressor).csv` and compared against `docs/dataset_description.pdf` and the official UCI Machine Learning Repository entry.

## Dataset Release Clarification (2020 vs Later Releases)

It is critical not to conflate the dataset used in this project with later MetroPT releases:

1. **MetroPT-3 (2020 Release, UCI Repository #791 — Used by MetroGuard):**
   - Temporal span: February 1, 2020 to September 1, 2020.
   - 1,516,948 observations.
   - Exactly 15 sensor features (7 analogue + 8 digital).
   - 4 air leak failure intervals documented in maintenance reports.
   - Contains no GPS coordinates or separate `Flowmeter` channel.
2. **MetroPT (2022 Zenodo Release / Nature Scientific Data Descriptor):**
   - Temporal span: January to June 2022 (a completely separate operational data collection period).
   - 10,979,547 observations recorded at 1 Hz.
   - 20 variables: 8 analogue (includes continuous `Flowmeter`), 8 digital, and 4 GPS channels (`gpsLong`, `gpsLat`, `gpsSpeed`, `gpsQuality`).
   - 3 catastrophic failures reported (Air leak on clients, Air leak on air dryer, Oil leak on compressor).

*MetroGuard strictly operates on the 2020 MetroPT-3 release (`data/raw/MetroPT3(AirCompressor).csv`). Citations or descriptions from the 2022 release are secondary and do not define the local dataset schema.*

## File Provenance & Identification

- **File Path:** `data/raw/MetroPT3(AirCompressor).csv`
- **File Size:** 218,300,507 bytes (~208.19 MB)
- **SHA-256 Hash:** `DB30CCB4EA402E3C8BF2C99DB06E288D4F2A772F6928F9DBE26A920D69793E24`
- **Local Companion Documentation:** `docs/dataset_description.pdf` (81,208 bytes)
- **Authoritative Repository Reference:** UCI Machine Learning Repository, Dataset #791, DOI `10.24432/C5VW3R`

## Schema & Sensor Attributes

The local CSV contains 17 columns: an unnamed row index (`column00`), an ISO datetime `timestamp`, and 15 sensor signals.

| Column | Inferred Type | Sensor Category | Physical Measurement | Range in CSV | Null Count |
|---|---|---|---|---|---|
| `column00` | BIGINT | Index | Serialized CSV row index | [0, 1516947] | 0 |
| `timestamp` | TIMESTAMP | Time | Datetime of observation | [2020-02-01 00:00:00, 2020-09-01 03:59:50] | 0 |
| `TP2` | DOUBLE | Analogue | Compressor pressure (bar) | [-0.032, 10.676] | 0 |
| `TP3` | DOUBLE | Analogue | Pneumatic panel pressure (bar) | [0.730, 10.302] | 0 |
| `H1` | DOUBLE | Analogue | Cyclonic separator filter drop (bar) | [-0.036, 10.288] | 0 |
| `DV_pressure` | DOUBLE | Analogue | Air dryer discharge pressure drop (bar) | [-0.032, 9.844] | 0 |
| `Reservoirs` | DOUBLE | Analogue | Downstream reservoir pressure (bar) | [0.712, 10.300] | 0 |
| `Oil_temperature` | DOUBLE | Analogue | Compressor oil temperature (°C) | [15.400, 89.050] | 0 |
| `Motor_current` | DOUBLE | Analogue | Motor electric current (A) | [0.020, 9.295] | 0 |
| `COMP` | DOUBLE | Digital | Intake valve electrical signal | {0.0, 1.0} | 0 |
| `DV_eletric` | DOUBLE | Digital | Outlet valve electrical signal | {0.0, 1.0} | 0 |
| `Towers` | DOUBLE | Digital | Active air dryer tower indicator | {0.0, 1.0} | 0 |
| `MPG` | DOUBLE | Digital | Intake activation under load (<8.2 bar) | {0.0, 1.0} | 0 |
| `LPS` | DOUBLE | Digital | Low pressure switch signal (<7 bar) | {0.0, 1.0} | 0 |
| `Pressure_switch` | DOUBLE | Digital | Pilot control valve discharge signal | {0.0, 1.0} | 0 |
| `Oil_level` | DOUBLE | Digital | Low oil level warning signal | {0.0, 1.0} | 0 |
| `Caudal_impulses` | DOUBLE | Digital | Flow pulse output signal | {0.0, 1.0} | 0 |

### Structural Observations:
- **Analogue Features (7):** `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current`. Small negative values on `TP2`, `H1`, `DV_pressure` (down to -0.036 bar) reflect sensor zero-calibration baselines near 0 bar.
- **Digital Signals (8):** `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses`. Every digital signal strictly assumes values in `{0.0, 1.0}` with zero invalid entries.
- **Artifact Column:** `column00` is an integer index from CSV serialization that must be omitted during ingestion.
- **Naming:** `DV_eletric` reflects the original source spelling.

## Sampling Cadence & Instance Count Reconciliation

The instance count and sampling cadence require formal reconciliation across sources:

```text
PUBLISHED/LOCAL DOCUMENTATION:
`docs/dataset_description.pdf` states:
"The dataset consists of 15169480 data points collected at 1Hz from February to August 2020 and is described by 15 features" (Page 2).
Table on Page 1 states: "Number of Instances: 15169480".

ACTUAL LOCAL DATA:
`data/raw/MetroPT3(AirCompressor).csv` contains exactly 1,516,948 rows.
Consecutive intervals (row_count - 1): exactly 1,516,947 intervals.
Empirical cadence distribution:
- Negative or zero intervals (diff <= 0s): 0 (0.000000%)
- Fast intervals (< 9.0s): 2 (0.000132%) [diff = 8.0s]
- Nominal cadence (9.0s - 13.0s): 1,516,580 (99.975807%)
  - 9.0s:  128,277 (8.456261%)
  - 10.0s: 1,337,521 (88.171901%)
  - 11.0s: 4,471 (0.294737%)
  - 12.0s: 38,321 (2.526192%)
  - 13.0s: 7,988 (0.526584%)
- Intermediate intervals (> 13.0s and <= 60.0s): 36 (0.002373%)
  - 14.0s: 3 | 15.0s: 1 | 17.0s: 3 | 18.0s: 1 | 19.0s: 5 | 20.0s: 3
  - 21.0s: 10 | 22.0s: 4 | 23.0s: 3 | 26.0s: 1 | 27.0s: 1 | 30.0s: 1
- Service breaks / gaps (> 60.0s): 331 (0.021820%)
  - Shortest gap > 60s: 104.0s (~1.73 minutes)
  - Largest gap: 172,918.0s (48.03 hours)
Total Reconciled: 1,516,580 + 36 + 331 = 1,516,947 intervals (100.000000%)

OFFICIAL UCI METADATA:
The official UCI Machine Learning Repository entry (#791, MetroPT-3) states:
- Number of Instances: 1,516,948
- Number of Features: 15
- Temporal span: February 2020 to August 2020.

The official UCI repository correctly reflects the actual artifact row count (1,516,948 rows). The string "15169480" in the companion PDF is an obvious typographical error (an appended trailing zero). While onboard hardware acquisition occurred at 1 Hz, the publicly distributed MetroPT-3 CSV is decimated to a nominal 10-second sampling interval (~0.1 Hz): cadence is predominantly 9–13 seconds (99.976%), a small number of larger sub-60-second intervals exist (36 intervals), and 331 service breaks have Δt > 60 seconds. The ~10-second cadence is an empirical property of the actual CSV and UCI metadata, not a claim from the original paper text. W represents observation count, not an elapsed-time duration.
```

## Chronology & Integrity Verification

- **Total Rows:** 1,516,948
- **Distinct Timestamps:** 1,516,948 (zero duplicate timestamps)
- **Duplicate Observations:** 0 across all 15 sensor features
- **Total Missing / Null Values:** 0 across all columns
- **Temporal Span:** 2020-02-01 00:00:00 to 2020-09-01 03:59:50 (213 calendar days)

### Monthly Breakdown

| Month | Observations | Timestamp Span | Failure Events Present |
|---|---|---|---|
| February 2020 | 214,850 | 2020-02-01 00:00:00 – 2020-02-28 23:57:08 | 0 (Clean normal) |
| March 2020 | 230,448 | 2020-03-01 04:00:09 – 2020-03-31 23:59:59 | 0 (Clean normal) |
| April 2020 | 198,734 | 2020-04-01 00:00:09 – 2020-04-30 23:39:23 | 1 (Event 1: Apr 18) |
| May 2020 | 212,800 | 2020-05-01 12:33:38 – 2020-05-31 23:49:44 | 1 (Event 2: May 29–30) |
| June 2020 | 216,514 | 2020-06-01 00:32:02 – 2020-06-30 23:59:58 | 1 (Event 3: Jun 5–7) |
| July 2020 | 222,638 | 2020-07-01 00:00:08 – 2020-07-31 23:59:52 | 1 (Event 4: Jul 15) |
| August 2020 | 220,434 | 2020-08-01 00:00:02 – 2020-08-31 20:15:10 | 0 |
| September 2020 | 530 | 2020-09-01 00:37:33 – 2020-09-01 03:59:50 | 0 |

## Documented Failure Events

The ground-truth failure intervals from transit maintenance logs in `docs/dataset_description.pdf` correspond to four air leak incidents:

| Event ID | Start Timestamp | End Timestamp | Duration | Rows in CSV | Maintenance Description |
|---|---|---|---|---|---|
| Event 1 | 2020-04-18 00:00:00 | 2020-04-18 23:59:59 | 24h 0m | 8,663 | Air leak on clients (pipe blowout; severe pressure drop) |
| Event 2 | 2020-05-29 23:30:00 | 2020-05-30 06:00:00 | 6h 30m | 2,360 | Air leak on air dryer (pilot valve malfunction; LPS triggers) |
| Event 3 | 2020-06-05 10:00:00 | 2020-06-07 14:30:00 | 52h 30m | 17,315 | Air leak (sustained pressure drops; maintenance 8-Jun) |
| Event 4 | 2020-07-15 14:30:00 | 2020-07-15 19:00:00 | 4h 30m | 1,622 | Air leak (maintenance 16-Jul 00:00) |

Total observations during active failure intervals: 29,960 out of 1,516,948 (1.975% of observations).
Ground truth is interval-based; there are no pointwise label columns in the raw data.
