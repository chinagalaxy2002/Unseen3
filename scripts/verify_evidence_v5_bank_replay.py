"""Real-data optimizer/replay test and independent frozen-localization check."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.moment_detr_gmr_evidence_v5.readouts import make_readout
from training.moment_detr_gmr_evidence_v5.feature_bank import Bank, load_fold_model
from training.moment_detr_gmr_evidence_v5.protocol import ROOT, EXPERIMENT, write_json
from training.moment_detr_gmr_evidence_v5.train_readout import predict
from training.moment_detr_gmr_evidence_v5.metrics import source_pair_metrics


def main():
    torch.set_num_threads(4)
    fold = "A1_action_01"
    bank = Bank(fold, "inner_train")
    ids = np.array([np.flatnonzero(bank.labels == 1)[0], np.flatnonzero(bank.labels == 0)[0]])
    cached = bank.batch(ids, "cpu", ["pooled", "slots", "base_logit", "query_tokens", "query_mask"])
    assert torch.allclose(cached["pooled"], cached["slots"].max(1).values, atol=0, rtol=0)
    baseline, opt, _ = load_fold_model(fold, "cpu")
    with torch.no_grad():
        replay = baseline.exist_head(cached["slots"])
    replay_error = float((replay - cached["base_logit"]).abs().max())
    if replay_error > 1e-5:
        raise RuntimeError("Cached head replay error")
    sys.path.insert(0, str(ROOT / "training/moment_detr_gmr"))
    from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
    from training.moment_detr_gmr.train import build_dataset_config
    config = build_dataset_config(opt, opt.train_path, keep_empty_gt=True)
    dataset = StartEndDataset(**config)
    metas, batch = start_end_collate([dataset[int(i)] for i in ids])
    inputs, _ = prepare_batch_inputs(batch, "cpu")
    with torch.no_grad():
        before = baseline(**inputs)
    model, _ = make_readout("R3")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(cached), torch.as_tensor(bank.labels[ids]))
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in baseline.parameters())
    optimizer.step()
    with torch.no_grad():
        after = baseline(**inputs)
    errors = {key: float((before[key] - after[key]).abs().max()) for key in ("pred_spans", "pred_logits")}
    assert all(v == 0 for v in errors.values())
    model.eval()
    output_before = model(cached).detach()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    clone, _ = make_readout("R3")
    clone.load_state_dict(torch.load(stream, weights_only=False))
    clone.eval()
    replay_delta = float((clone(cached) - output_before).abs().max())
    assert replay_delta == 0
    video_only, video_keys = make_readout("Cv")
    video_scores = predict(video_only, bank, video_keys, "cpu")
    source_control = source_pair_metrics(bank.rows, video_scores)
    assert source_control["pair_count"] > 0 and source_control["pair_acc"] == 0.5
    write_json(EXPERIMENT / "bank_replay_verification.json", {"passed": True, "fold": fold,
               "real_positive_and_negative_rows": True, "main_loss_finite": float(loss.detach()),
               "cached_pooling_exact": True, "cached_head_replay_maxerr": replay_error,
               "independent_localization_before_after_optimizer": errors,
               "baseline_parameters_have_no_gradients": True, "readout_checkpoint_replay_maxerr": replay_delta,
               "video_only_source_pair_acc_exactly_half": source_control,
               "opened_roles": ["inner_train"], "novel_dev_read": False})
    print("PASS", errors, "head replay", replay_error)


if __name__ == "__main__":
    main()
