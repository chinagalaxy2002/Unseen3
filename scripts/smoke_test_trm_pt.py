"""
Smoke Test for Moment-DETR-TRM-PT Branch [PAPER-DERIVED-REPRODUCTION].
Verifies dataset loading, model forward, auxiliary pseudo-span loss, and backward pass.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
gmr_training_path = str(REPO_ROOT / "training" / "moment_detr_gmr")
if gmr_training_path not in sys.path:
    sys.path.insert(0, gmr_training_path)

import torch
import torch.nn as nn

def run_smoke_test_pt():
    print("=" * 60)
    print("STARTING MOMENT-DETR-TRM-PT SMOKE TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # A. Import Test
    print("\n--- [Step A] Python Import Test ---")
    from models.moment_detr_trm_pt.moment_detr_trm_pt import MomentDETR_TRM_PT, build_moment_detr_trm_pt
    from models.moment_detr_trm_pt.trm_pt_loss import SetCriterionTRM_PT, build_criterion_trm_pt
    from training.moment_detr_trm_pt.dataset_trm_pt import (
        StartEndDatasetTRM_PT,
        start_end_collate_trm_pt,
        prepare_batch_inputs_trm_pt,
    )
    from training.moment_detr_trm.config import BaseOptionsTRM
    from training.moment_detr_trm.train_trm import build_dataset_config_trm
    print("[+] Successfully imported TRM-PT modules.")

    # B. Dataset Test
    print("\n--- [Step B] Dataset Smoke Test ---")
    opt_mgr = BaseOptionsTRM("moment_detr_trm", "charades_sta_semantic_novelty", "clip_slowfast")
    opt_mgr.parse()
    opt = opt_mgr.option
    opt.device = str(device)
    opt.phrase_feat_dir = str(REPO_ROOT / "features" / "phrase_data" / "clip_phrase")
    opt.pt_pseudo_dir = str(REPO_ROOT / "features" / "phrase_data" / "pt_pseudo_labels")
    opt.v_feat_dirs = [
        str(REPO_ROOT / "features" / "charades_video" / "vid_slowfast"),
        str(REPO_ROOT / "features" / "charades_video" / "vid_clip"),
    ]
    opt.t_feat_dir = str(REPO_ROOT / "features" / "semantic_existence_v2" / "shared_clip_text")
    opt.train_path = str(REPO_ROOT / "data" / "release" / "semantic_existence_v2" / "A1" / "train.jsonl")

    dset_config = build_dataset_config_trm(opt, opt.train_path, load_labels=True, keep_empty_gt=False, partition_filter="S+")
    dset_config["pt_pseudo_dir"] = opt.pt_pseudo_dir
    dataset = StartEndDatasetTRM_PT(**dset_config)
    print(f"[+] TRM-PT dataset loaded. Samples count: {len(dataset)}")

    raw_samples = [dataset[i] for i in range(4)]
    for i, s in enumerate(raw_samples):
        m_in = s["model_inputs"]
        assert "phrase_pseudo_spans" in m_in, f"Missing phrase_pseudo_spans in sample {i}"
        assert "phrase_pseudo_conf" in m_in, f"Missing phrase_pseudo_conf in sample {i}"
        assert m_in["phrase_pseudo_spans"].shape == (10, 2)
        assert m_in["phrase_pseudo_conf"].shape == (10,)

    batch_meta, batched_inputs = start_end_collate_trm_pt(raw_samples)
    model_inputs, targets = prepare_batch_inputs_trm_pt(batched_inputs, device)
    print(f"[+] Batched inputs prepared with phrase_pseudo_spans: {targets['phrase_pseudo_spans'].shape}")

    # C. Model Forward Test
    print("\n--- [Step C] Model Forward Test ---")
    model = build_moment_detr_trm_pt(opt).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**model_inputs)

    assert "pred_phrase_spans" in outputs, "Missing pred_phrase_spans in outputs"
    print(f"[+] pred_phrase_spans shape: {outputs['pred_phrase_spans'].shape} (expected [4, 10, 2])")
    assert outputs["pred_phrase_spans"].shape == (4, 10, 2)
    assert (outputs["pred_phrase_spans"] >= 0.0).all() and (outputs["pred_phrase_spans"] <= 1.0).all()

    # D. Loss Test
    print("\n--- [Step D] Loss Test ---")
    opt.lambda_pt_span = 1.0
    opt.lambda_pt_giou = 0.5
    criterion = build_criterion_trm_pt(opt).to(device)

    loss_dict = criterion(outputs, targets)
    print("[+] Computed TRM-PT loss dictionary:")
    for k, v in loss_dict.items():
        print(f"    {k}: {float(v):.4f}")
        assert not torch.isnan(v), f"Loss {k} is NaN"

    assert "loss_phrase_pseudo_span" in loss_dict
    assert "loss_phrase_pseudo_giou" in loss_dict

    # E. Backward & Gradient Flow Test
    print("\n--- [Step E] Backward & Gradient Flow Test ---")
    model.train()
    criterion.train()

    train_outputs = model(**model_inputs)
    train_loss_dict = criterion(train_outputs, targets)
    total_loss = sum(
        train_loss_dict[k] * criterion.weight_dict[k]
        for k in train_loss_dict.keys()
        if k in criterion.weight_dict
    )

    model.zero_grad()
    total_loss.backward()

    head_grad = model.phrase_span_head.layers[-1].weight.grad
    assert head_grad is not None, "phrase_span_head gradient is None"
    grad_norm = head_grad.norm().item()
    assert grad_norm > 0, "phrase_span_head gradient is 0"
    print(f"[+] phrase_span_head grad_norm = {grad_norm:.6f}")

    # F. Tiny Overfit Test
    print("\n--- [Step F] 10-Iteration Overfit Test ---")
    overfit_opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    init_loss = None
    last_loss = None
    for it in range(10):
        overfit_opt.zero_grad()
        out = model(**model_inputs)
        l_dict = criterion(out, targets)
        l_tot = sum(l_dict[k] * criterion.weight_dict[k] for k in l_dict if k in criterion.weight_dict)
        l_tot.backward()
        overfit_opt.step()
        val = l_tot.item()
        if it == 0:
            init_loss = val
        last_loss = val

    assert last_loss < init_loss, f"Loss did not decrease: init={init_loss:.4f}, last={last_loss:.4f}"
    print(f"[+] TRM-PT overfit test PASSED: {init_loss:.4f} -> {last_loss:.4f}")

    print("\n" + "=" * 60)
    print("ALL TRM-PT SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test_pt()
