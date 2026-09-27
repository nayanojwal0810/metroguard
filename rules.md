# MetroGuard — Operating Rules

## Authority

Use this order when interpreting project instructions:

1. Scientific integrity and leakage safety.
2. `execution_plan.md` for project intent and scope.
3. `decision_log.md` for approved technical decisions.
4. `rules.md` for implementation behavior.
5. `python_design.md`, `document_design.md`, `colab.md`, and `git_workflow.md` for implementation-specific rules.
6. Repository code and artifacts as evidence of what actually exists.

If these sources conflict, stop and request a decision. Do not silently choose.

## Role model

**ChatGPT**
- owns technical direction and methodology;
- designs/reviews experiments;
- reviews evidence and claims;
- approves major decisions.

**Antigravity**
- inspects and implements;
- creates code, tests, configs, and documents;
- runs lightweight validation;
- reports evidence;
- stops when a decision or heavy computation is required.

**User**
- operates the local environment;
- runs heavy/long commands when requested;
- runs Colab workloads;
- returns outputs;
- approves major direction changes.

## Mandatory execution behavior

Before changing anything:

1. Inspect the current repository.
2. Read the active project-control documents.
3. Identify the exact current objective.
4. State the files to be created/modified and why.
5. Implement only the approved work.
6. Run relevant lightweight validation.
7. Report exactly what changed and what passed/failed.

Do not start unrelated work.

## Stop conditions

Stop and return control to ChatGPT when:

- the dataset materially differs from documented assumptions;
- a source conflict changes the baseline;
- a leakage risk is found;
- evaluation definitions need to change;
- the split needs to change;
- a major architecture choice is required;
- an approved method is not feasible;
- results materially contradict the current hypothesis;
- an experiment requires protected-holdout access for tuning;
- scope expansion is proposed;
- a heavy command exceeds practical local limits.

Use:

```text
PROBLEM:
EVIDENCE:
IMPACT:
OPTIONS:
RECOMMENDATION:
DECISION NEEDED FROM CHATGPT:
```

## Heavy-runtime rule

Estimate runtime and resource requirements before execution.

If a command may be expensive or long-running:

1. Do not start it.
2. Give the exact command for the User to run.
3. State what output/files must be returned.
4. Stop.

Never poll, loop, repeatedly retry, or wait for the command.

Respect CPU, RAM, disk, Colab Free runtime, GPU availability, and quota limits.

Reuse cached artifacts when valid.

## Scientific integrity

Never:

- tune on final holdout results;
- change split boundaries to improve metrics;
- hide failed experiments;
- delete disappointing results;
- claim improvement without evidence;
- use random temporal splits without explicit approval;
- change metric definitions after seeing results;
- treat quality rejection as free performance.

When results are weak, diagnose the cause and preserve the evidence.

## Repository discipline

Keep the repository clean.

Use only necessary files. Avoid:

- junk files;
- duplicate scripts;
- debug dumps;
- temporary exports;
- unnecessary notebooks;
- generated caches in Git;
- monolithic scripts.

Check repository cleanliness before and after significant work.

## Git rule

Git is the synchronization boundary between CPU and Colab.

There is one canonical codebase. CPU and Colab are execution environments, not separate implementations.

Do not maintain independent `src` copies.

Follow `git_workflow.md`.

## Communication

Use short, direct, human-readable updates.

At the start of meaningful work state:

- objective;
- what will be done;
- important decision;
- expected output.

At completion state:

- what changed;
- key result;
- failures/limitations;
- current gate status;
- next decision.

Do not use motivational filler.

## Approval rule

Antigravity may make ordinary refactoring decisions when scientific meaning is unchanged.

ChatGPT approval is required for:

- model changes;
- feature changes with scientific impact;
- split/evaluation changes;
- threshold/policy changes;
- regime definitions;
- quality-gate logic;
- metric definitions;
- major dependency or architecture changes;
- production lifecycle changes;
- scope changes.
