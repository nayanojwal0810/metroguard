# MetroGuard — Git Integration Workflow

## Principle

Git is the integration boundary between the local CPU environment and Colab.

There is one canonical MetroGuard codebase.

CPU and Colab are execution environments, not independent projects.

## Source of truth

Git-controlled:

- Python source;
- tests;
- configs;
- documentation;
- experiment definitions;
- compact metrics/results;
- reproducibility manifests.

Usually not Git-controlled:

- raw dataset;
- large processed datasets;
- large checkpoints;
- Colab caches;
- temporary files;
- local virtual environments.

## Normal development flow

```text
ChatGPT approves work
    ↓
Antigravity inspects repository
    ↓
Antigravity implements on CPU
    ↓
lightweight validation
    ↓
commit
    ↓
push
```

## Heavy experiment flow

```text
approved repository commit
    ↓
Colab pull
    ↓
heavy run
    ↓
save outputs
    ↓
record provenance
    ↓
commit approved results/metadata
    ↓
push
    ↓
CPU pull
    ↓
review evidence
```

## No concurrent source edits

While a heavy Colab experiment is running, do not modify the experiment's core source/configuration.

Freeze the relevant Git state until the run result is captured.

If source changes are necessary, stop the experiment workflow and obtain a new approved commit.

## Branching

Keep the default workflow simple.

Use the canonical branch for stable project state.

A temporary experiment branch may be used when a run needs isolated result commits, but it must merge back through normal Git review.

Do not create branches merely for routine edits.

## Commit quality

Commits should represent coherent changes.

Good:

```text
add causal window validation
freeze alert evaluation definitions
add regime calibration
```

Avoid:

```text
update
changes
stuff
final_final
```

Never commit generated junk merely to make Git appear active.

## Before push

Check:

- `git status`;
- unintended files;
- dataset/checkpoint files;
- test status;
- documentation consistency;
- experiment identity.

## Before heavy Colab execution

Record the exact commit being executed.

Do not rely on “latest code” without identifying the commit.

## After Colab execution

The result must identify:

- Git commit used;
- configuration;
- dataset identity;
- experiment ID;
- output artifacts.

If source changes were needed after the run, do not rewrite the experiment result. Create a new experiment record.

## Conflict handling

If CPU and Colab changed the same source file:

1. stop;
2. do not blindly merge;
3. preserve both versions;
4. report the conflict;
5. request technical direction.

Results must never overwrite historical evidence.

## Clean repository rule

Run a cleanliness check before meaningful commits.

The repository should not accumulate:

- ZIP bundles;
- copied datasets;
- cache folders;
- temporary logs;
- duplicate outputs;
- editor files.
