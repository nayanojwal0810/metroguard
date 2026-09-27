# MetroGuard — Decision Log

This file records approved technical decisions that affect scientific meaning, evaluation, architecture, or reproducibility.

Do not silently rewrite historical decisions.

## Decision format

### DEC-XXX — [Decision title]

**Date:**  
**Status:** Proposed / Approved / Superseded

**Decision**

[What was decided.]

**Reason**

[Why it was chosen.]

**Evidence**

[Experiment, source, metric, artifact, or assumption.]

**Alternatives**

[Important alternatives considered.]

**Impact**

[What this decision changes.]

**Approved by**

ChatGPT / User

---

## Initial decisions

### DEC-001 — Project objective

**Status:** Approved

**Decision**

MetroGuard focuses on turning compressor anomaly scores into a defensible maintenance-alert decision.

**Reason**

This preserves the original anomaly-detection formulation while creating a focused engineering contribution.

**Evidence**

Approved project execution specification.

**Impact**

New neural architectures are outside the current core scope unless later approved.

**Approved by**

ChatGPT / User

### DEC-002 — Primary dataset

**Status:** Approved

**Decision**

Use the UCI MetroPT-3 dataset as the primary dataset.

**Reason**

It provides real compressor telemetry and documented failure intervals suitable for anomaly-detection evaluation.

**Impact**

Dataset version and hash must be recorded before model evaluation.

**Approved by**

ChatGPT / User

### DEC-003 — Baseline formulation

**Status:** Approved

**Decision**

Use a sparse autoencoder as the primary research baseline, reproduced from the selected source as faithfully as practical.

**Impact**

Baseline assumptions must be documented before modifications are introduced.

**Approved by**

ChatGPT / User

### DEC-004 — Protected evaluation

**Status:** Approved

**Decision**

Use chronological training → calibration → final holdout separation.

**Impact**

The final holdout cannot be used for tuning.

**Approved by**

ChatGPT / User

### DEC-005 — MetroGuard modifications

**Status:** Approved

**Decision**

Evaluate:
- cost-aware alert policy;
- operating-regime-aware policy;
- input-quality gate.

**Impact**

These components must be tested individually and together through controlled ablations.

**Approved by**

ChatGPT / User

### DEC-006 — CPU/Colab integration

**Status:** Approved

**Decision**

Git is the synchronization boundary. One canonical codebase is shared across CPU and Colab.

**Impact**

No independent Colab source tree is permitted.

**Approved by**

ChatGPT / User

---

## Reserved decisions

Do not fill these from assumption. Record them only after evidence is available.

- dataset hash/version;
- exact feature set;
- causal aggregation/window size;
- train/calibration/holdout boundaries;
- baseline architecture details;
- baseline threshold method;
- alert episode semantics;
- cost objective and weights;
- regime definition;
- quality-gate thresholds;
- monitoring thresholds;
- retraining triggers;
- serving/tracking stack.
