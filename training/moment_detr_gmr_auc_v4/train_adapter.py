from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.moment_detr_gmr_auc_v4.common import ROOT, SPLITS, auc, finite_or_raise, make_adapter, write_json
from training.moment_detr_gmr_auc_v4.pair_sampler import PairSampler, ranking_losses


def load_bank(path):
    x = np.load(path, allow_pickle=False)
    return {k: x[k] for k in x.files}


def diagnostic(scores, bank):
    ret = {"pooled_auroc": auc(bank["exist_label"], scores)}
    for key, name in (("action_id", "action"), ("composition_id", "composition")):
        vals = []
        groups = np.unique(bank[key])
        for g in groups:
            m = bank[key] == g
            v = auc(bank["exist_label"][m], scores[m])
            if np.isfinite(v): vals.append(v)
        ret[f"{name}_macro_auroc"] = float(np.mean(vals)) if vals else None
        ret[f"worst_{name}_auroc"] = float(np.min(vals)) if vals else None
    return ret


def train(split, device):
    out = ROOT / "results" / "moment_detr_gmr_auc_v4" / split
    tr, va = load_bank(out / "feature_bank_train.npz"), load_bank(out / "feature_bank_val.npz")
    if np.any(np.char.startswith(tr["partition"].astype(str), "U")) or np.any(np.char.startswith(va["partition"].astype(str), "U")):
        raise RuntimeError("U rows found in adapter training or validation bank")
    n = len(tr["qid"])
    if len(set(tr["qid"].tolist())) != n or len(set(va["qid"].tolist())) != len(va["qid"]):
        raise RuntimeError("duplicate qids in feature bank")
    if n != len(tr["exist_label"]): raise RuntimeError("train bank count mismatch")
    torch.manual_seed(3407); np.random.seed(3407)
    z = torch.as_tensor(tr["base_exist_repr"], dtype=torch.float32, device=device)
    s0 = torch.as_tensor(tr["base_exist_logit"], dtype=torch.float32, device=device)
    y = torch.as_tensor(tr["exist_label"], dtype=torch.float32, device=device)
    zv = torch.as_tensor(va["base_exist_repr"], dtype=torch.float32, device=device)
    s0v = torch.as_tensor(va["base_exist_logit"], dtype=torch.float32, device=device)
    adapter = make_adapter(z.shape[1], device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    baseline_val = auc(va["exist_label"], va["base_exist_logit"])
    best_auc, best_epoch = baseline_val, 0
    best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    history = [{"epoch": 0, "val": diagnostic(va["base_exist_logit"], va), "selected": True}]
    total_steps = math.ceil(n / 256)
    audit = {"exact_query": 0, "composition": 0, "action": 0, "unpaired_positive": int((tr["exist_label"] == 1).sum())}
    global_cross_semantic = 0
    global_pairs = 0
    replacements = 0

    for epoch in range(1, 51):
        adapter.train()
        sampler = PairSampler(tr, seed=3407 + epoch, same_pairs=256)
        audit["unpaired_positive"] = sampler.unpaired
        permutation = np.random.default_rng(3407 + epoch).permutation(n)
        level_counts = np.zeros(3, dtype=np.int64)
        epoch_losses = np.zeros(4, dtype=np.float64)
        for step in range(total_steps):
            idx = permutation[step * 256:(step + 1) * 256]
            ip, inn, rep = sampler.global_indices(); replacements += int(rep)
            isp, isn, levels = sampler.semantic_indices()
            optimizer.zero_grad(set_to_none=True)
            # Three independent streams share only the adapter parameters.
            s_bce, d_bce = adapter(z[idx], s0[idx])
            l_bce = F.binary_cross_entropy_with_logits(s_bce, y[idx])
            # Score all AUC stream items in a single batched adapter call.
            gidx = np.concatenate([ip, inn])
            sg, _ = adapter(z[gidx], s0[gidx])
            l_global, _ = ranking_losses(sg, np.arange(128), np.arange(128, 256), [], [])
            if len(isp):
                sidx = np.concatenate([isp, isn])
                ss, _ = adapter(z[sidx], s0[sidx])
                _, l_same = ranking_losses(ss, [], [], np.arange(len(isp)), np.arange(len(isp), 2 * len(isp)))
                level_counts += np.bincount(levels, minlength=3)
            else:
                l_same = s_bce.sum() * 0.0
            l_anchor = (s_bce - s0[idx]).square().mean()
            loss = l_bce + l_global + l_same + 0.1 * l_anchor
            finite_or_raise(loss, l_bce, l_global, l_same, l_anchor, d_bce)
            loss.backward()
            optimizer.step()
            epoch_losses += np.asarray([float(loss.detach()), float(l_bce.detach()), float(l_global.detach()), float(l_same.detach())])
            global_cross_semantic += int(np.sum(tr["composition_id"][ip][:, None] != tr["composition_id"][inn][None, :]))
            global_pairs += 128 * 128
        adapter.eval()
        with torch.no_grad():
            sv, dv = adapter(zv, s0v)
        finite_or_raise(sv, dv)
        val_scores = sv.cpu().numpy().astype(np.float64)
        val_auc = auc(va["exist_label"], val_scores)
        selected = val_auc > best_auc + 1e-15
        if selected:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        history.append({"epoch": epoch, "val": diagnostic(val_scores, va), "mean_total_loss": float(epoch_losses[0] / total_steps),
                        "mean_bce": float(epoch_losses[1] / total_steps), "mean_global_auc": float(epoch_losses[2] / total_steps),
                        "mean_same": float(epoch_losses[3] / total_steps), "same_pair_level_counts": level_counts.tolist(), "selected": selected})
        print(f"{split} epoch={epoch:02d}/50 val_pooled_auc={val_auc:.8f} best={best_auc:.8f}@{best_epoch}", flush=True)

    adapter.load_state_dict(best_state)
    ckpt = {"adapter": best_state, "best_epoch": int(best_epoch), "best_seen_val_auroc": float(best_auc),
            "baseline_seen_val_auroc": float(baseline_val), "seed": 3407,
            "architecture": "LayerNorm(D+1)-Linear64-GELU-Dropout0.1-Linear1",
            "loss": {"bce": 1.0, "global_auc": 1.0, "same_semantic": 1.0, "anchor": 0.1},
            "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
            "epochs": 50, "batch_size_bce": 256, "steps_per_epoch": total_steps, "residual_bound": 2.0}
    torch.save(ckpt, out / "best.ckpt")
    same_total = int(sum(history[-1]["same_pair_level_counts"])) if history else 0
    pair_summary = {"exact_query": float(level_counts[0] / level_counts.sum()) if level_counts.sum() else 0.0,
                    "composition": float(level_counts[1] / level_counts.sum()) if level_counts.sum() else 0.0,
                    "action": float(level_counts[2] / level_counts.sum()) if level_counts.sum() else 0.0,
                    "unpaired_positive_count": int(sampler.unpaired)}
    write_json(out / "adapter_training_history.json", {"split": split, "baseline_seen_val_auroc": baseline_val,
                "best_seen_val_auroc": best_auc, "best_epoch": best_epoch, "history": history,
                "same_semantic_pair_fractions_last_epoch": pair_summary,
                "global_pair_cross_composition_fraction": global_cross_semantic / max(1, global_pairs),
                "replacement_steps": replacements, "feature_bank_train_count": n, "feature_bank_val_count": len(va["qid"])})
    print(f"{split} selected best_epoch={best_epoch} Seen-val AUROC={best_auc:.8f}; epoch0={baseline_val:.8f}", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", choices=SPLITS, required=True); p.add_argument("--device", default="cuda:0")
    a = p.parse_args(); train(a.split, a.device)


if __name__ == "__main__": main()
