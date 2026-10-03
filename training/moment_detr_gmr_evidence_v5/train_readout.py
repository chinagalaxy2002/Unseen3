"""Seen-trained readouts; novel dev is never opened by this training module."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.moment_detr_gmr_evidence_v5.readouts import make_readout, parameter_count
from training.moment_detr_gmr_evidence_v5.feature_bank import Bank
from training.moment_detr_gmr_evidence_v5.metrics import auc
from training.moment_detr_gmr_evidence_v5.protocol import EXPERIMENT, RESULTS, digest, read_rows, write_json


@torch.no_grad()
def predict(model, bank, keys, device):
    model.eval()
    values = []
    indices_all = np.asarray(list(bank.vid_to_index.values())) if keys == ["video_mean"] else np.arange(len(bank.rows))
    for start in range(0, len(indices_all), 256):
        indices = indices_all[start:start + 256]
        values.append(model(bank.batch(indices, device, keys)).cpu().numpy())
    scores = np.concatenate(values).astype(np.float64)
    if keys == ["video_mean"]:
        by_video = {str(bank.rows[int(i)]["vid"]): float(score) for i, score in zip(indices_all, scores)}
        scores = np.asarray([by_video[str(r["vid"])] for r in bank.rows], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("Nonfinite predictions")
    return scores


def valid_pair_indices(bank, fold):
    path = EXPERIMENT / "inner_folds" / fold / "inner_train_source_pairs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    pairs = []
    for r in rows:
        if r["pair_eligible"]:
            pairs.append((bank.qid_to_index[str(r["positive_qid"])], bank.qid_to_index[str(r["negative_qid"])]))
    if not pairs:
        raise RuntimeError("No eligible source pairs")
    return np.asarray(pairs, dtype=np.int64)


def train(fold, variant, device="cuda:0", seed=3407, pair_weight=0.0, epochs=50, protocol=None):
    if epochs != 50:
        raise ValueError("Frozen readout protocol requires 50 complete epochs")
    protocol_path = EXPERIMENT / "READOUT_FREEZE.json" if protocol is None else Path(protocol).resolve()
    freeze = json.loads(protocol_path.read_text())
    if variant not in freeze["variants"] or fold not in freeze["folds"]:
        raise ValueError("Variant/fold not in frozen readout protocol")
    if seed not in freeze["seeds"] or pair_weight not in freeze["pair_weights"]:
        raise ValueError("Seed/pair weight not in frozen protocol")
    for path, checksum in freeze["code_hashes"].items():
        from training.moment_detr_gmr_evidence_v5.protocol import ROOT
        if digest(ROOT / path) != checksum:
            raise RuntimeError(f"Changed frozen readout code: {path}")
    if pair_weight != 0.0 and variant.startswith("D_"):
        raise ValueError("Residual diagnostic already uses its inherited ranking objectives")
    stage = "S1_pair" if pair_weight else "P3_bce"
    output = RESULTS / "readouts" / fold / f"{variant}_{stage}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.ckpt").exists() or (output / "history.json").exists():
        raise RuntimeError(f"Existing readout artifacts: {output}")
    tr, va = Bank(fold, "inner_train"), Bank(fold, "inner_seen_val")
    # Do not instantiate or read the novel-development bank here.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model, keys = make_readout(variant)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    diagnostic_mode = variant.startswith("D_")
    anchor_weight = 0.0 if variant == "D_no_anchor" else 0.1
    batch_size = 256 if diagnostic_mode else 32
    base_auc = auc(va.labels, va.arrays["base_logit"])
    best_auc, best_epoch = -float("inf"), None
    best_state, history = None, []
    baseline_is_epoch_zero = diagnostic_mode
    if diagnostic_mode:
        best_auc, best_epoch = base_auc, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pair_indices = valid_pair_indices(tr, fold) if pair_weight else None
    if diagnostic_mode:
        from training.moment_detr_gmr_auc_v4.pair_sampler import PairSampler
        from training.moment_detr_gmr_evidence_v5.protocol import semantic
        sampler_bank = {"exist_label": tr.labels.astype(np.int8),
                        "normalized_query": np.asarray([" ".join(r["query"].lower().split()) for r in tr.rows]),
                        "composition_id": np.asarray([semantic(r, "composition") for r in tr.rows]),
                        "action_id": np.asarray([semantic(r, "action") for r in tr.rows])}
    started = time.time()
    status = {"fold": fold, "variant": variant, "seed": seed, "stage": stage,
              "status": "running", "epochs_planned": epochs, "pair_weight": pair_weight,
              "trainable_parameters": parameter_count(model), "baseline_is_epoch_zero": baseline_is_epoch_zero,
              "selection": "inner Seen-val pooled AUROC; earliest tie",
              "freeze_hash": digest(protocol_path), "baseline_val_auroc": base_auc,
              "novel_dev_read_during_training": False}
    write_json(output / "status.json", status)
    try:
        for epoch in range(1, epochs + 1):
            rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(tr.rows))
            covered = np.zeros(len(tr.rows), dtype=np.int8)
            loss_sum, count, norms = 0.0, 0, []
            model.train()
            if diagnostic_mode:
                sampler = PairSampler(sampler_bank, seed=seed + epoch, same_pairs=256)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                covered[indices] += 1
                batch = tr.batch(indices, device, keys)
                scores = model(batch)
                labels = torch.as_tensor(tr.labels[indices], device=device)
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                if pair_weight:
                    selected = pair_indices[rng.integers(0, len(pair_indices), size=32)]
                    pair_scores = model(tr.batch(selected.reshape(-1), device, keys)).reshape(-1, 2)
                    loss = loss + pair_weight * F.softplus(1 - pair_scores[:, 0] + pair_scores[:, 1]).mean()
                if diagnostic_mode:
                    positive, negative, _ = sampler.global_indices()
                    global_scores = model(tr.batch(np.concatenate([positive, negative]), device, keys))
                    loss = loss + F.softplus(1 - global_scores[:128, None] + global_scores[None, 128:]).mean()
                    sp, sn, _ = sampler.semantic_indices()
                    if len(sp):
                        semantic_scores = model(tr.batch(np.concatenate([sp, sn]), device, keys))
                        loss = loss + F.softplus(1 - semantic_scores[:len(sp)] + semantic_scores[len(sp):]).mean()
                    loss = loss + anchor_weight * (scores - batch["base_logit"]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # V4 diagnostic inherits its original no-clipping optimizer.
                if not diagnostic_mode:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm):
                        raise RuntimeError("Nonfinite gradient norm")
                    norms.append(float(norm))
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
                count += len(indices)
            if not np.all(covered == 1):
                raise RuntimeError("Main training rows not covered exactly once")
            val_scores = predict(model, va, keys, device)
            val_auc = auc(va.labels, val_scores)
            if val_auc > best_auc + 1e-15:
                best_auc, best_epoch = val_auc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({"epoch": epoch, "mean_loss": loss_sum / count, "seen_val_auroc": val_auc,
                            "main_unique_coverage": count, "mean_gradient_norm": float(np.mean(norms)) if norms else None})
            write_json(output / "history.json", history)
            status.update(epochs_completed=epoch, best_seen_val_auroc=best_auc, best_epoch=best_epoch,
                          elapsed_seconds=time.time() - started)
            write_json(output / "status.json", status)
            print(f"{fold} {variant} epoch={epoch:02d}/50 SeenAUROC={val_auc:.6f} best={best_auc:.6f}@{best_epoch}", flush=True)
        deployment = "learned" if best_auc > base_auc + 1e-15 else "baseline_fallback"
        torch.save({"model": best_state, "variant": variant, "keys": keys, "seed": seed,
                    "best_epoch": best_epoch, "best_seen_val_auroc": best_auc, "baseline_val_auroc": base_auc,
                    "pair_weight": pair_weight, "deployment": deployment, "epochs": epochs,
                    "bank_train_identity": tr.metadata["identity"], "bank_val_identity": va.metadata["identity"],
                    "freeze_hash": digest(protocol_path), "freeze_path": str(protocol_path)}, output / "best.ckpt")
        status.update(status="selected", deployment=deployment, checkpoint_hash=digest(output / "best.ckpt"),
                      elapsed_seconds=time.time() - started)
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--pair-weight", type=float, default=0.0)
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args()
    train(args.fold, args.variant, args.device, args.seed, args.pair_weight, protocol=args.protocol)
