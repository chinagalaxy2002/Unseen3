# Moment-DETR-TRM Generalization Project

This directory contains the documentation, source audits, specifications, and experimental results for integrating **Phrase-Level Temporal Relationship Mining (TRM / TRM-PT)** with **Moment-DETR** under the Charades-STA Semantic Novelty benchmark (`semantic_existence_v2`).

---

## File Index

- [`TASK.md`](TASK.md): Live task status and evidence tracker.
- [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json): Provenance, git commit hashes, and SHA256 checksums of core files.
- [`BASELINE_AUDIT.md`](BASELINE_AUDIT.md): Detailed architectural and data interface audit of Moment-DETR.
- [`TRM_SOURCE_AUDIT.md`](TRM_SOURCE_AUDIT.md): Audit of official TRM code (`minghangz/TRM`), mechanism analysis, and paper-repo comparison.
- [`METHOD_SPEC.md`](METHOD_SPEC.md): Formal mapping specification from TRM to Moment-DETR and loss formulation.
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md): Execution stages, frozen hyperparameters, and evaluation plan.
- [`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md): Step-by-step implementation log and milestone checks.
