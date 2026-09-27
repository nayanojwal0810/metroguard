# MetroGuard — Colab Execution Rules

## Purpose

Google Colab is an execution environment for workloads that are too expensive or slow for the local CPU environment.

The repository remains the single source of truth.

## Core rule

Do not create a second codebase in Colab.

Colab must execute the same repository code used by CPU.

Do not edit core source code in Colab unless explicitly approved and synchronized through Git.

## Heavy-run workflow

```text
Approved Git commit
    ↓
Colab pulls that commit
    ↓
Install exact dependencies
    ↓
Verify dataset and environment
    ↓
Run the approved command
    ↓
Save experiment outputs
    ↓
Record Git commit + config + dataset identity
    ↓
Commit approved lightweight results/metadata
    ↓
Push
    ↓
CPU environment pulls results
    ↓
ChatGPT reviews evidence
```

## Before a run

Verify:

- Git commit;
- configuration;
- dataset presence/version;
- Python version;
- GPU availability if required;
- available disk;
- expected runtime.

Do not start a different experiment because a GPU is available.

## GPU rule

Use GPU only when it materially reduces runtime for the approved workload.

CPU is preferred for:

- data inspection;
- validation;
- unit tests;
- leakage tests;
- small experiments;
- inference checks;
- documentation.

GPU may be used for:

- approved deep-learning training;
- approved hyperparameter searches;
- expensive final experiments.

## Dataset

The raw MetroPT-3 CSV must not be committed to Git.

Recommended location in the user-managed workspace:

```text
data/
└── raw/
    └── MetroPT3(AirCompressor).csv
```

Record file size and SHA-256 where practical.

Do not silently replace the dataset with another copy or release.

## Reproducibility

Every heavy run should record:

- experiment ID;
- Git commit;
- configuration;
- dataset identity/hash;
- environment information;
- seed;
- command;
- output artifact locations.

## Output handling

Keep large files outside Git unless explicitly approved.

Examples:

- raw datasets;
- large checkpoints;
- temporary caches;
- large processed arrays.

Git may contain compact reproducibility metadata, metrics tables, manifests, and small plots when useful.

## Quota discipline

Do not run broad hyperparameter searches on Colab Free without approval.

Prefer:

1. small smoke run;
2. narrow targeted experiment;
3. approved full run.

Reuse valid artifacts.

Never poll for completion.

## User handoff

When a heavy command is required, provide exactly:

```text
Command:
<copy/paste command>

Expected outputs:
<files or summary>

Return:
<paste the terminal output and list the generated files>
```

Then stop.

## Colab safety

Do not:

- alter evaluation splits;
- tune on final holdout results;
- overwrite the only copy of a model;
- delete previous experiment evidence;
- save outputs without experiment identity.

## Recovery

If Colab disconnects:

- inspect existing outputs;
- reuse completed artifacts;
- resume only from a documented state;
- do not blindly restart a full expensive run.

If the run fails materially, return the error and context rather than repeatedly retrying.
