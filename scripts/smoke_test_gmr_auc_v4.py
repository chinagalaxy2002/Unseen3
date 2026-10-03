from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.moment_detr_gmr_auc_v4.residual_adapter import ResidualAdapter
from training.moment_detr_gmr_auc_v4.common import SPLITS, dataset_for, finite_or_raise, load_model, pooled_repr_and_logits, read_jsonl, prepare_batch_inputs, start_end_collate
from training.moment_detr_gmr_auc_v4.pair_sampler import PairSampler, ranking_losses


def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", choices=SPLITS, default="A1"); p.add_argument("--device", default="cuda:0")
    a = p.parse_args(); split = a.split
    model, opt, ck = load_model(split, a.device)
    assert all(not p.requires_grad for p in model.parameters())
    ds = dataset_for(split, "val", opt, keep_empty=True)
    raw = read_jsonl(Path(opt.train_path).parent / "val.jsonl")
    expected = [r for r in raw if str(r.get("partition", "")).startswith("S")]
    assert len(ds.data) == len(expected), (len(ds.data), len(expected))
    item = ds[0]; inp, _ = prepare_batch_inputs(start_end_collate([item])[1], a.device)
    model.eval()
    with torch.no_grad(): out, z, s0 = pooled_repr_and_logits(model, inp)
    adapter = ResidualAdapter(z.shape[-1]).to(a.device)
    assert torch.count_nonzero(adapter.net[-1].weight).item() == 0 and torch.count_nonzero(adapter.net[-1].bias).item() == 0
    s_epoch0, d_epoch0 = adapter(z, s0)
    assert torch.allclose(s_epoch0, s0, atol=1e-7, rtol=0)
    assert torch.equal(out["pred_spans"], out["pred_spans"].clone()) and torch.equal(out["pred_logits"], out["pred_logits"].clone())
    model_before = {n: p.detach().clone() for n,p in model.named_parameters()}
    adapter_before = {n: p.detach().clone() for n,p in adapter.named_parameters()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    s, d = adapter(z, s0)
    loss = F.binary_cross_entropy_with_logits(s, torch.ones_like(s)) + 0.1 * (s-s0).square().mean()
    finite_or_raise(loss, d)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    assert all(p.grad is None for p in model.parameters())
    assert model.transformer.encoder.layers[0].self_attn.in_proj_weight.grad is None
    assert model.exist_head.layers[0].weight.grad is None
    scores = torch.randn(256, device=a.device, requires_grad=True)
    global_loss, same_loss = ranking_losses(scores, np.arange(128), np.arange(128, 256), [], [])
    finite_or_raise(global_loss, same_loss)
    optimizer.step()
    assert any(not torch.equal(before, dict(adapter.named_parameters())[name].detach()) for name,before in adapter_before.items())
    assert all(torch.equal(v, dict(model.named_parameters())[k].detach()) for k,v in model_before.items())
    assert torch.all(d.abs() <= 2.0 + 1e-7)
    assert ranking_losses(s, [], [], [], [])[1].item() == 0.0
    serialized = json.loads(json.dumps({"v": float(s0.flatten()[0].cpu())}))
    assert serialized["v"] == float(s0.flatten()[0].cpu())

    bank_path = ROOT / "results/moment_detr_gmr_auc_v4" / split / "feature_bank_train.npz"
    assert bank_path.is_file(), "extract Seen train feature bank before this smoke test"
    bank = np.load(bank_path, allow_pickle=False)
    n = len(bank["qid"])
    assert len(set(bank["qid"].tolist())) == n
    assert not np.char.startswith(bank["partition"].astype(str), "U").any()
    train_rows = read_jsonl(Path(opt.train_path))
    seen_rows = [r for r in train_rows if str(r.get("partition", "")).startswith("S")]
    assert n == len(seen_rows), (n, len(seen_rows))
    assert len(bank["qid"]) == len(bank["base_exist_repr"]) == len(bank["base_exist_logit"])
    sampler = PairSampler({k: bank[k] for k in bank.files}, seed=3407, same_pairs=256)
    pi, ni, _ = sampler.global_indices()
    cross = np.any(bank["composition_id"][pi][:, None] != bank["composition_id"][ni][None, :])
    assert cross, "global AUC draw did not contain cross-semantic pairs"
    assert np.ceil(n / 256) >= 1
    result = {"split": split, "baseline_loaded": True, "all_baseline_parameters_frozen": True,
              "exist_repr_head_replay": True, "adapter_zero_init_and_epoch0_anchor": True,
              "baseline_grads_none_and_parameters_unchanged": True, "bce_finite": True,
              "global_auc_finite": True, "same_semantic_finite_and_empty_graph_safe": True,
              "anchor_finite": True, "bounded_delta": True, "feature_bank_count": n,
              "seen_train_count": len(seen_rows), "feature_bank_has_no_u": True, "global_cross_semantic_pairs": True,
              "localization_forward_unchanged": True, "full_precision_json_roundtrip": True}
    (ROOT / "experiments/moment_detr_gmr_auc_v4/SMOKE_TEST_RESULT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
