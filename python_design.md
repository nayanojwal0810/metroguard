# MetroGuard — Python Design Rules

## General

Write production-style Python that is small, readable, testable, and deterministic.

Use snake_case for Python filenames.

Prefer modular design over large scripts.

Use OOP when state, lifecycle, configuration, or component ownership makes a class genuinely useful. Do not force classes around simple pure transformations or metrics.

## Structure

Prefer:

- one clear responsibility per module;
- small focused functions;
- small cohesive classes;
- explicit inputs/outputs;
- reusable utilities;
- no hidden global state.

Avoid:

- monolithic scripts;
- giant manager classes;
- duplicated business logic;
- deeply nested control flow;
- magic constants.

## Types

Use type hints for public functions, class interfaces, and important internal boundaries.

Prefer explicit data structures over loosely shaped dictionaries when a stable schema is needed.

## Configuration

Do not hard-code experiment decisions that should be configurable.

Keep experiment parameters in versioned configuration files where appropriate.

Record the effective configuration with experiment artifacts.

## Logging

Use professional logging.

Preferred:

```text
[INFO] Baseline training completed. Reconstruction loss: ...
[INFO] Holdout evaluation completed. False alarms/day: ...
[WARNING] 14% of input windows failed the quality contract.
[ERROR] Model artifact could not be loaded.
```

Avoid casual output such as:

```text
done
yay
test passed!!!
```

Do not print sensitive or unnecessarily large data.

## Docstrings and comments

Public modules/classes/functions should have crisp docstrings describing purpose and important assumptions.

Use comments only when they clarify non-obvious reasoning, leakage constraints, or implementation trade-offs.

Do not write comments that simply repeat the code.

## ML-specific rules

Make data boundaries explicit.

A function that fits anything must make its fitting data source clear.

Causal feature/window functions must make temporal behavior obvious.

Model code must not know hidden evaluation labels.

Threshold/policy code must be separable from model inference.

Metrics code must use explicit definitions and testable inputs.

## Determinism

Where practical:

- set random seeds;
- record seeds;
- use deterministic configuration where supported;
- record library/runtime versions.

Do not claim exact reproducibility when the underlying environment cannot guarantee it.

## Testing

Important pure logic must have unit tests.

Required testing areas include, as applicable:

- schema validation;
- timestamp handling;
- causal windows;
- split boundaries;
- leakage checks;
- anomaly score calculation;
- alert policy;
- regime assignment;
- quality gate;
- event/lead-time metrics;
- serving contract.

## Performance

Avoid unnecessary full-dataset scans.

Do not repeatedly reload large CSV files when cached processed artifacts can be reused.

Prefer streaming/chunked processing or efficient formats when justified by actual resource limits.

Do not optimize prematurely.

## Dependency discipline

Add a dependency only when it solves a real project need.

Keep the dependency list small and documented.

## Definition of clean code

Code is acceptable when another engineer can understand:

- what it does;
- what data it consumes;
- what assumptions it makes;
- what it returns;
- how it is tested;
- how it is configured.
