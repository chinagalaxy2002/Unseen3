"""Check invariance, masking, gradients, identity rejection and statistical clusters."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.moment_detr_gmr_evidence_v5.readouts import make_readout, parameter_count
from training.moment_detr_gmr_evidence_v5.metrics import cluster_delta_interval
from training.moment_detr_gmr_evidence_v5.protocol import EXPERIMENT, write_json


def main():
    torch.manual_seed(3407)
    torch.set_num_threads(4)
    batch = {"pooled": torch.randn(4, 256), "slots": torch.randn(4, 10, 256),
             "query_tokens": torch.randn(4, 32, 512), "query_mask": torch.cat([torch.ones(4, 7), torch.zeros(4, 25)], 1),
             "video_mean": torch.randn(4, 2816), "base_logit": torch.randn(4)}
    parameters = {}
    for name in ("R1", "R1_matched", "R2", "R3", "Cq", "Cv", "D_v4", "D_no_bound", "D_no_anchor"):
        model, _ = make_readout(name)
        parameters[name] = parameter_count(model)
        model.eval()
        scores = model(batch)
        assert scores.shape == (4,) and torch.isfinite(scores).all()
        if name.startswith("D_"):
            assert torch.equal(scores, batch["base_logit"])
        if name in ("R2", "R3", "Cq"):
            masked = {**batch, "query_tokens": batch["query_tokens"].clone()}
            masked["query_tokens"][:, 7:] = torch.randn(4, 25, 512) * 1000
            assert torch.allclose(scores, model(masked), atol=1e-6, rtol=0), "Padding changed scores"
        if name == "R3":
            shuffled = {**batch, "slots": batch["slots"][:, torch.randperm(10)]}
            assert torch.allclose(scores, model(shuffled), atol=1e-6, rtol=0), "Slot order changed set readout"
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), torch.tensor([1., 0., 1., 0.]))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert parameters["R2"] == parameters["R3"]
    assert abs(parameters["R1_matched"] / parameters["R2"] - 1) < .2
    rows = [{"vid": "v1", "exist_label": 1}, {"vid": "v1", "exist_label": 0},
            {"vid": "v2", "exist_label": 1}, {"vid": "v2", "exist_label": 0}]
    same = np.array([.8, .2, .7, .1])
    interval = cluster_delta_interval(rows, same, same, replicates=30)
    assert interval["delta_ci95"] == [0., 0.]
    write_json(EXPERIMENT / "readout_verification.json", {"passed": True, "parameters": parameters,
               "slot_permutation_invariant": True, "padding_invariant": True,
               "zero_init_replays_baseline": True, "gradient_step_finite": True,
               "R2_R3_same_parameter_count": True, "R1_matched_within_20_percent": True,
               "paired_video_bootstrap_identity": True,
               "scope": "readout semantics and cluster behavior; real bank replay audited separately"})
    print("PASS", parameters)


if __name__ == "__main__":
    main()
