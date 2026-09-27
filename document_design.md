# MetroGuard — Document Design Rules

## Purpose

Project documents must make the work easy to understand, reproduce, review, and defend.

Write for a technical reader who has not seen the earlier chat.

## Style

Use:

- concise headings;
- short paragraphs;
- compact tables;
- direct English;
- concrete evidence;
- consistent terminology.

Avoid:

- motivational language;
- repetitive summaries;
- artificial verbosity;
- marketing language;
- unnecessary diagrams;
- long explanations of obvious code.

## GitHub-facing language

Do not use labels such as:

- Phase 1;
- Step 1;
- Checkpoint 1;
- Task completed;
- AI status update.

Use professional engineering terms such as:

- Current scope;
- Evaluation design;
- Implementation status;
- Validation;
- Decision;
- Result;
- Limitation.

Internal control logic may exist in `current_state.md`, but public-facing project documents should read like normal engineering documentation.

## Evidence rules

Never write a result before it exists.

Every important quantitative claim should identify:

- metric;
- evaluation population/window;
- relevant model/policy;
- whether the result is calibration or holdout evidence.

Do not write “improved” unless the comparison and evidence support it.

Use exact terminology for:

- alert;
- event;
- early detection;
- late detection;
- false alarm;
- data-quality rejection;
- holdout.

## Result documents

Prefer:

```text
Objective
Method
Evaluation
Results
Interpretation
Limitations
Decision
```

## Technical reports

A report should be understandable without source code.

Include only information necessary to explain:

- problem;
- data;
- methodology;
- evaluation;
- results;
- operational implications;
- limitations;
- reproducibility.

## README

README is the public entry point.

It should eventually answer:

- What is MetroGuard?
- Why does it exist?
- How does the system work?
- What was evaluated?
- What were the actual results?
- What are the limitations?
- How can it be reproduced?

Keep implementation-control instructions out of the README.

## Claims

Do not use unsupported terms such as:

- SOTA;
- production-ready;
- real-time;
- hallucination-free;
- failure-proof;
- safety-certified.

Use precise alternatives tied to evidence.

## Failed experiments

Do not hide scientifically relevant failures.

Summarize them concisely when they inform the final design or limitation.

## Formatting

Use Markdown consistently.

Prefer tables for compact comparisons.

Prefer code blocks for commands and schemas.

Do not create documentation only to increase file count.

Every document must have a clear purpose and owner.
