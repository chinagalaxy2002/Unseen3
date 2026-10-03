"""Inspect feature headers and Seen durations without changing feature mapping."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.moment_detr_gmr_evidence_v5.protocol import EXPERIMENT, RELEASE, read_rows, write_json


def header(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("features.npy") as stream:
            version = np.lib.format.read_magic(stream)
            shape, _, _ = np.lib.format._read_array_header(stream, version)
        return shape, archive.namelist()


def main():
    baseline = json.loads((EXPERIMENT.parent / "moment_detr_gmr_auc_v4/baseline_audit.json").read_text())
    # Config comes from the recorded canonical baseline, not directory labels.
    import torch
    from training.moment_detr_gmr_evidence_v5.protocol import BASELINES
    opt = torch.load(BASELINES / "A1/moment/best.ckpt", map_location="cpu", weights_only=False)["opt"]
    videos = {}
    for split in ("A1", "C1"):
        for part in ("train", "val"):
            for r in read_rows(RELEASE / split / (part + ".jsonl")):
                if r["partition"].startswith("S"):
                    if r["vid"] in videos and abs(videos[r["vid"]]["duration"] - r["duration"]) > 1e-5:
                        raise RuntimeError("Inconsistent video duration")
                    videos.setdefault(r["vid"], {"duration": r["duration"], "max_gt_end": 0})
                    videos[r["vid"]]["max_gt_end"] = max([videos[r["vid"]]["max_gt_end"]] + [w[1] for w in r.get("relevant_windows", [])])
    offsets, deltas, truncated, beyond = [], [], 0, 0
    entries = {}
    for vid, meta in sorted(videos.items()):
        shapes, keys = zip(*(header(Path(d) / (vid + ".npz")) for d in opt.v_feat_dirs))
        raw_length = min(shape[0] for shape in shapes)
        length = min(raw_length, opt.max_v_l)
        offsets.append(length * opt.clip_length - meta["duration"])
        deltas.append(abs(shapes[0][0] - shapes[1][0]))
        truncated += raw_length > opt.max_v_l
        beyond += meta["max_gt_end"] > length * opt.clip_length
        entries[vid] = dict(meta, modality_shapes=[list(s) for s in shapes], effective_length=length,
                            nominal_visible_end=length * opt.clip_length,
                            embedded_timestamps=any("time" in key for modality_keys in keys for key in modality_keys))
    report = {"video_count": len(videos), "feature_order": list(opt.v_feat_dirs),
              "checkpoint_clip_length": opt.clip_length, "max_v_l": opt.max_v_l,
              "modality_length_difference_max": int(max(deltas)), "truncated_video_count": truncated,
              "nominal_end_minus_duration_quantiles": np.quantile(offsets, [0, .25, .5, .75, 1]).tolist(),
              "videos_with_gt_beyond_nominal_feature_end": beyond,
              "videos_with_embedded_timestamps": sum(x["embedded_timestamps"] for x in entries.values()),
              "baseline_mapping": "unchanged: GT normalized by T*clip_length; predictions scaled by duration",
              "local_roi_gate": "pending extraction timestamp provenance; headers alone do not establish exact times",
              "videos": entries}
    write_json(EXPERIMENT / "time_grid_audit.json", report)
    print({k: v for k, v in report.items() if k != "videos"})


if __name__ == "__main__":
    main()
