# MetroGuard — Execution Plan

## Project

**MetroGuard — Operational Early-Warning System for Metro Train Compressor Failures**

MetroGuard is a production-style ML engineering project for turning compressor anomaly scores into defensible maintenance-alert decisions.

The project is based on the Davari et al. DSAA 2021 sparse-autoencoder formulation and the UCI MetroPT-3 dataset.

## Objective

Build and evaluate a leakage-safe time-series anomaly detection system that:

1. reproduces the research baseline faithfully;
2. converts anomaly scores into an operational alert decision;
3. tests cost-aware alerting;
4. tests a small operating-regime-aware policy;
5. rejects or degrades gracefully on poor-quality telemetry;
6. supports monitoring, retraining-candidate generation, versioning, promotion, and rollback.

The project is a research/portfolio demonstration. It is not a railway safety controller and must not be presented as one.

## Technical boundary

### Baseline

Sensor telemetry → causal preprocessing/windowing → sparse autoencoder → reconstruction error → fixed/global threshold → alert.

### MetroGuard

Sensor telemetry → input-quality validation → causal preprocessing/windowing → sparse autoencoder → anomaly score → operating-regime determination → cost-aware alert policy → operational state.

Operational states must include at least:

- `NORMAL`
- `ALERT`
- `DATA_QUALITY_ISSUE`

## Research foundation

Primary research:

Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2021), *Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry*, DSAA 2021.

Primary dataset:

UCI MetroPT-3 Dataset, DOI `10.24432/C5VW3R`.

The UCI page currently reports 1,516,948 observations, 15 features, and documented compressor failure intervals. Published metadata also contains conflicting cadence wording; the implementation must verify actual timestamp spacing from the downloaded CSV instead of assuming a fixed frequency.

## Contribution boundary

The project contribution is the **decision layer and its evaluation**, not a new neural-network architecture.

Infrastructure such as FastAPI, Docker, experiment tracking, dashboards, tests, monitoring, or model packaging supports the engineering story but is not itself the research contribution.

Do not claim:

- a new anomaly-detection algorithm;
- a novel neural architecture;
- statistical generalization to all metro trains;
- safety-critical deployment;
- hallucination-free or failure-proof behavior;
- production railway deployment.

## Required evaluation design

Use one immutable chronological dataset version and a strict:

**training → calibration → final holdout**

design.

Training data may fit:

- preprocessing parameters;
- anomaly model;
- normal representation.

Calibration data may fit:

- thresholds;
- alert-policy parameters;
- cost weights;
- regime definitions;
- non-test decision rules.

The final holdout is for final evaluation only. Once final holdout evaluation begins, no tuning is permitted from its results.

For every inference decision at time `t`, only information available at or before `t` may influence:

- preprocessing;
- features/windows;
- anomaly score;
- regime;
- policy;
- quality state.

## Mandatory leakage controls

Never:

- random-split adjacent time-series rows/windows;
- fit preprocessing on future data;
- use centered rolling features;
- interpolate from future observations;
- use post-event information in pre-event features;
- tune against the final holdout;
- define regimes using future labels or future-derived information;
- silently allow windows to cross evaluation boundaries;
- create online decisions using future labels.

Leakage tests are mandatory before trusting any improvement.

## Alert semantics

Before threshold or policy optimization, freeze the evaluation definitions for:

- one alert episode;
- alert merging;
- cooldown;
- event matching;
- early detection window;
- late detection;
- lead-time calculation;
- time in alert;
- false-alarm counting;
- quality-gated periods.

These definitions must remain unchanged across controlled comparisons.

## Core metrics

Headline metrics:

- false alarms per day;
- early-event recall;
- detection lead time;
- time in alert.

Secondary metrics where meaningful:

- failure-window PR-AUC;
- anomaly/reconstruction-error distributions;
- regime-level alert rates;
- alert-duration distributions;
- data-quality rejection rate;
- inference latency.

Because the dataset contains only a small number of documented events, report every event individually and avoid unsupported significance claims.

## Controlled experiment ladder

The main ablation should be:

| ID | System |
|---|---|
| B0 | Baseline |
| B1 | Baseline + cost-aware alert policy |
| B2 | Baseline + regime-aware policy |
| B3 | Baseline + cost-aware + regime-aware policy |
| B4 | B3 + input-quality gate = MetroGuard |

All comparisons should use the same:

- dataset version;
- chronological boundaries;
- model inputs;
- preprocessing boundary;
- anomaly model;
- causal window rules;
- protected holdout.

Only the intended decision-layer component may change in an ablation.

A mixed or negative result is acceptable if it is measured, explained, and documented honestly.

## Cost-aware policy

Use a transparent proxy objective covering the operational trade-off between:

- unnecessary alert burden;
- useful early detection;
- alert occupancy.

Do not claim true monetary optimization unless real monetary costs are available.

All assumptions and weights must be configurable and recorded.

## Regime-aware policy

Use a small, interpretable set of operating regimes that is observable at inference time.

Rules:

- few regimes;
- no future information;
- no post-failure information;
- calibration only from permitted data;
- fallback to a global policy when regime evidence is insufficient;
- report regime coverage and rare-regime behavior.

## Input-quality gate

Validate, where justified:

- required columns;
- data types;
- missingness;
- timestamp validity;
- cadence/coverage;
- physical/range constraints;
- insufficient history;
- validated operating-envelope violations.

A rejected period must be visible in reporting. Quality rejection must never be treated as successful anomaly suppression.

Always report quality rejection rate and whether known events occurred during rejected periods.

## Production-style lifecycle

The validated system should support:

Data → validation → causal preprocessing → model inference → decision policy → alert/event record → monitoring → retraining candidate → validation → versioned release → rollback.

Every component must have a documented lifecycle purpose.

## Reproducibility

Record:

- Python/dependency versions;
- Git commit;
- dataset version/hash where possible;
- configuration version;
- random seed where relevant;
- exact split boundaries;
- model/policy version;
- commands used;
- expected artifacts.

Large raw data and model binaries must not be committed to Git unless explicitly approved.

## Evidence standard

No technical claim is final until supported by:

- an experiment;
- a metric;
- an artifact;
- a reproducible run; or
- an explicitly stated assumption.

Resume wording is written only after final evidence exists.

## Completion criteria

The project is complete when:

- the baseline is reproducible;
- chronology and leakage controls are validated;
- alert semantics are frozen;
- controlled ablations are complete;
- the final holdout remains protected;
- event-level and operational metrics are reported;
- quality-gate behavior is measured;
- production-style lifecycle components are demonstrated where justified;
- results and limitations are reproducible;
- README/report/interview material matches actual evidence.
