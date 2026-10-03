"""Atomic partial/final P3 tables. Does not change queues or select checkpoints."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.moment_detr_gmr_evidence_v5.protocol import EXPERIMENT, RESULTS


def atomic_text(path, content):
    fd, name = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        stream.write(content)
    os.replace(name, path)


def main():
    freeze = json.loads((EXPERIMENT / "READOUT_FREEZE.json").read_text())
    rows = []
    for path in sorted((RESULTS / "readouts").glob("*/*_P3_bce_seed3407/result.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            # The other device may still be writing a result; the next table refresh includes it.
            continue
    expected = len(freeze["folds"]) * len(freeze["variants"])
    macros = {}
    lines = ["# P3 inner readout results", "", "Seen-only inner development; formal U was not read.", "",
             f"Completed {len(rows)}/{expected} runs. Partial results do not change the frozen queue.", "",
             "| Fold | Variant | Epoch | Seen base | Seen learned | Novel base | Novel learned | ΔNovel | Deployment |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['fold']} | {r['variant']} | {r['best_epoch']} | {r['baseline']['seen']['pooled_auroc']:.5f} | {r['learned']['seen']['pooled_auroc']:.5f} | {r['baseline']['novel']['pooled_auroc']:.5f} | {r['learned']['novel']['pooled_auroc']:.5f} | {r['delta_novel_auroc']:+.5f} | {r['deployment']} |")
    lines.extend(["", "| Variant | Folds | Macro ΔNovel | Macro ΔSeen | Positive novel folds | Parameters |", "|---|---:|---:|---:|---:|---:|"])
    for variant in freeze["variants"]:
        subset = [r for r in rows if r["variant"] == variant]
        if not subset:
            continue
        macros[variant] = {"folds_completed": len(subset), "complete": len(subset) == len(freeze["folds"]),
                           "macro_delta_novel": float(np.mean([r["delta_novel_auroc"] for r in subset])),
                           "macro_delta_seen": float(np.mean([r["delta_seen_auroc"] for r in subset])),
                           "positive_novel_folds": sum(r["delta_novel_auroc"] > 0 for r in subset)}
        m = macros[variant]
        lines.append(f"| {variant} | {len(subset)}/4 | {m['macro_delta_novel']:+.5f} | {m['macro_delta_seen']:+.5f} | {m['positive_novel_folds']} | {subset[0]['trainable_parameters']} |")
    lines.extend(["", "R2 and R3 have exactly equal parameter counts. R1_matched controls pooled readout capacity.",
                  "Learned and baseline-fallback policies are reported separately; fallback does not erase failed learned models.",
                  "R4 is pending temporal provenance; no conclusion about local-video evidence follows from this table.",
                  "These are seed-3407 development results, not the three-seed G2 gate or confirmatory evidence.", ""])
    summary = {"completed": len(rows), "expected": expected, "status": "completed" if len(rows) == expected else "partial",
               "macros": macros, "results": rows, "local_roi_gate": "pending"}
    atomic_text(EXPERIMENT / "P3_READOUT_RESULTS.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    atomic_text(EXPERIMENT / "P3_READOUT_RESULTS.md", "\n".join(lines))
    print(f"P3: {len(rows)}/{expected} results")


if __name__ == "__main__":
    main()
