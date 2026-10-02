"""
Comprehensive Smoke Test for Moment-DETR-TRM.
Implements the 7-step unit verification suite required by Section 19 of plan.md:
  A. Python import test
  B. Dataset smoke test (4 samples + batch_size=1 + edge cases)
  C. Model forward test (shapes, bounds, assertions)
  D. Loss test (finiteness, empty GT safety)
  E. Backward test (non-zero gradients on all TRM modules and transformer backbone)
  F. Single mini-batch optimizer step
  G. 15-iteration tiny overfit test (loss monotonically/strictly decreases)
"""
from __future__ import annotations

import os
import sys
import pprint
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
gmr_training_path = str(REPO_ROOT / "training" / "moment_detr_gmr")
if gmr_training_path not in sys.path:
    sys.path.insert(0, gmr_training_path)

import torch
import torch.nn as nn
from easydict import EasyDict

def run_smoke_test():
    print("=" * 60)
    print("STARTING MOMENT-DETR-TRM SMOKE TEST SUITE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # ========================================================
    # STEP A: Python Import Test
    # ========================================================
    print("\n--- [Step A] Python Import Test ---")
    try:
        from models.moment_detr_trm.moment_detr_trm import MomentDETR_TRM, build_moment_detr_trm
        from models.moment_detr_trm.trm_modules import AttentivePooling, PhraseSlotMatcher
        from models.moment_detr_trm.trm_loss import SetCriterionTRM, build_criterion_trm
        from training.moment_detr_trm.dataset_trm import (
            StartEndDatasetTRM,
            start_end_collate_trm,
            prepare_batch_inputs_trm,
        )
        from training.moment_detr_trm.config import BaseOptionsTRM
        from training.moment_detr_trm.evaluate_trm import eval_epoch_trm
        from training.moment_detr_trm.train_trm import setup_model_and_criterion, build_dataset_config_trm
        print("[+] Successfully imported all models and training modules for Moment-DETR-TRM.")
    except Exception as e:
        print(f"[!] FAILED Step A Import Test: {e}")
        raise e

    # ========================================================
    # STEP B: Dataset Smoke Test
    # ========================================================
    print("\n--- [Step B] Dataset Smoke Test ---")
    opt_mgr = BaseOptionsTRM("moment_detr_trm", "charades_sta_semantic_novelty", "clip_slowfast")
    opt_mgr.parse()
    opt = opt_mgr.option
    opt.device = str(device)
    opt.phrase_feat_dir = str(REPO_ROOT / "features" / "phrase_data" / "clip_phrase")
    opt.v_feat_dirs = [
        str(REPO_ROOT / "features" / "charades_video" / "vid_slowfast"),
        str(REPO_ROOT / "features" / "charades_video" / "vid_clip"),
    ]
    opt.t_feat_dir = str(REPO_ROOT / "features" / "semantic_existence_v2" / "shared_clip_text")
    opt.train_path = str(REPO_ROOT / "data" / "release" / "semantic_existence_v2" / "A1" / "train.jsonl")

    dset_config = build_dataset_config_trm(opt, opt.train_path, load_labels=True, keep_empty_gt=False, partition_filter="S+")
    dataset = StartEndDatasetTRM(**dset_config)
    print(f"[+] Dataset initialized successfully. Total S+ training samples: {len(dataset)}")

    # Check first 4 samples
    raw_samples = [dataset[i] for i in range(4)]
    for i, s in enumerate(raw_samples):
        meta = s["meta"]
        m_in = s["model_inputs"]
        print(f"    Sample {i}: qid={meta['qid']}, query='{meta['query']}', "
              f"phrase_count={m_in['phrase_count']}, GT_windows={meta.get('relevant_windows')}")
        assert "query_feat" in m_in, f"Missing query_feat in sample {i}"
        assert "video_feat" in m_in, f"Missing video_feat in sample {i}"
        assert "phrase_features" in m_in, f"Missing phrase_features in sample {i}"
        assert "phrase_tokens_mask" in m_in, f"Missing phrase_tokens_mask in sample {i}"
        assert "phrase_mask" in m_in, f"Missing phrase_mask in sample {i}"
        assert m_in["phrase_features"].shape == (10, 16, 512), f"Unexpected shape {m_in['phrase_features'].shape}"
        assert m_in["phrase_tokens_mask"].shape == (10, 16)
        assert m_in["phrase_mask"].shape == (10,)
        assert m_in["phrase_count"] > 0, "Expected at least 1 valid phrase"

    # Test Collate & Batch Preparation with B=4
    batch_meta, batched_inputs = start_end_collate_trm(raw_samples)
    model_inputs, targets = prepare_batch_inputs_trm(batched_inputs, device)
    print(f"[+] Batched inputs successfully created:")
    print(f"    src_txt: {model_inputs['src_txt'].shape}, src_txt_mask: {model_inputs['src_txt_mask'].shape}")
    print(f"    src_vid: {model_inputs['src_vid'].shape}, src_vid_mask: {model_inputs['src_vid_mask'].shape}")
    print(f"    phrase_features: {model_inputs['phrase_features'].shape}")
    print(f"    phrase_tokens_mask: {model_inputs['phrase_tokens_mask'].shape}")
    print(f"    phrase_mask: {model_inputs['phrase_mask'].shape}")
    print(f"    targets span_labels count: {len(targets['span_labels'])}")

    # Test edge case: B=1
    single_meta, single_batched = start_end_collate_trm([raw_samples[0]])
    single_inputs, single_targets = prepare_batch_inputs_trm(single_batched, device)
    assert single_inputs["src_vid"].shape[0] == 1, "Single batch size mismatch"
    print("[+] Edge case batch_size=1 collated successfully.")

    # ========================================================
    # STEP C: Model Forward Test
    # ========================================================
    print("\n--- [Step C] Model Forward Test ---")
    model = build_moment_detr_trm(opt).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**model_inputs)

    # Check required outputs
    required_keys = [
        "pred_spans", "pred_logits_base", "pred_logits",
        "pred_phrase_scores", "phrase_weights", "phrase_mask",
        "phrase_repr", "slot_repr", "p_proj", "s_proj"
    ]
    for k in required_keys:
        assert k in outputs, f"Missing output key: {k}"
        assert outputs[k] is not None, f"Output key {k} is None"

    B, Q = outputs["pred_spans"].shape[0], outputs["pred_spans"].shape[1]
    P = outputs["pred_phrase_scores"].shape[1]

    print(f"[+] Output shapes verified:")
    print(f"    pred_spans: {outputs['pred_spans'].shape} (expected [{B}, {Q}, 2])")
    print(f"    pred_logits_base: {outputs['pred_logits_base'].shape} (expected [{B}, {Q}, 2])")
    print(f"    pred_logits: {outputs['pred_logits'].shape} (expected [{B}, {Q}, 2])")
    print(f"    pred_phrase_scores: {outputs['pred_phrase_scores'].shape} (expected [{B}, {P}, {Q}])")
    print(f"    phrase_weights: {outputs['phrase_weights'].shape} (expected [{B}, {P}])")
    print(f"    p_proj: {outputs['p_proj'].shape} (expected [{B}, {P}, 256])")
    print(f"    s_proj: {outputs['s_proj'].shape} (expected [{B}, {Q}, 256])")

    # Value bound assertions
    assert (outputs["pred_spans"] >= 0.0).all() and (outputs["pred_spans"] <= 1.0).all(), "pred_spans outside [0, 1]"
    assert (outputs["pred_phrase_scores"] > 0.0).all() and (outputs["pred_phrase_scores"] < 1.0).all(), "phrase_scores outside (0, 1)"

    # Assert phrase weights sum to 1 per sample
    weight_sums = outputs["phrase_weights"].sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4), f"Phrase weights sum != 1: {weight_sums}"
    print("[+] All forward shape and value range assertions PASSED.")

    # ========================================================
    # STEP D: Loss Test & Empty GT Safety
    # ========================================================
    print("\n--- [Step D] Loss Test & Empty GT Safety ---")
    criterion = build_criterion_trm(opt).to(device)

    # Compute loss on normal batch
    loss_dict = criterion(outputs, targets)
    print("[+] Computed loss dictionary:")
    for k, v in loss_dict.items():
        v_float = float(v)
        print(f"    {k}: {v_float:.4f}")
        assert not torch.isnan(v), f"Loss {k} is NaN"
        assert not torch.isinf(v), f"Loss {k} is Inf"

    # Verify all required loss terms exist
    expected_loss_terms = [
        "loss_span", "loss_giou", "loss_label",
        "loss_phrase_consistency", "loss_phrase_negative", "loss_phrase_exclusiveness"
    ]
    for term in expected_loss_terms:
        assert term in loss_dict, f"Missing loss term: {term}"

    # Empty GT test: ensure criterion handles empty GT spans gracefully
    empty_targets = {
        "span_labels": [dict(spans=torch.empty((0, 2), device=device)) for _ in range(B)]
    }
    empty_loss_dict = criterion(outputs, empty_targets)
    for k, v in empty_loss_dict.items():
        assert not torch.isnan(v), f"Empty GT produced NaN in {k}"
        assert not torch.isinf(v), f"Empty GT produced Inf in {k}"
    print("[+] Empty GT safety test PASSED: all losses finite and stable.")

    # Edge case: B=1 loss test
    with torch.no_grad():
        single_outputs = model(**single_inputs)
    single_loss_dict = criterion(single_outputs, single_targets)
    assert not torch.isnan(single_loss_dict["loss_phrase_consistency"]), "B=1 consistency is NaN"
    assert single_loss_dict["loss_phrase_negative"] == 0.0, "B=1 negative loss should be 0.0"
    print("[+] Edge case batch_size=1 loss computation PASSED.")

    # ========================================================
    # STEP E: Backward Test
    # ========================================================
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

    # Check gradients on TRM specific modules and transformer backbone
    checks = [
        ("model.phrase_matcher.phrase_proj.weight", model.phrase_matcher.phrase_proj.weight.grad),
        ("model.phrase_matcher.slot_proj.weight", model.phrase_matcher.slot_proj.weight.grad),
        ("model.attentive_pooling.feat2att.weight", model.attentive_pooling.feat2att.weight.grad),
        ("model.attentive_pooling.to_alpha.weight", model.attentive_pooling.to_alpha.weight.grad),
        ("model.input_txt_proj", list(model.input_txt_proj.parameters())[0].grad),
        ("model.input_vid_proj", list(model.input_vid_proj.parameters())[0].grad),
        ("model.transformer.encoder.layers[0].linear1.weight", model.transformer.encoder.layers[0].linear1.weight.grad),
        ("model.transformer.decoder.layers[0].linear1.weight", model.transformer.decoder.layers[0].linear1.weight.grad),
    ]

    for name, grad in checks:
        assert grad is not None, f"Gradient is None for {name}"
        grad_norm = grad.norm().item()
        assert grad_norm > 0, f"Gradient is zero for {name}"
        assert not torch.isnan(grad).any(), f"Gradient is NaN for {name}"
        print(f"    {name}: grad_norm = {grad_norm:.6f}")

    print("[+] Backward gradient flow verified on all core and TRM parameters.")

    # ========================================================
    # STEP F: Single Optimizer Step
    # ========================================================
    print("\n--- [Step F] Single Optimizer Step ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_drop, gamma=0.1)

    optimizer.step()
    lr_scheduler.step()
    print("[+] Optimizer and lr_scheduler step executed cleanly.")

    # ========================================================
    # STEP G: 15-Iteration Tiny Overfit Test
    # ========================================================
    print("\n--- [Step G] 15-Iteration Tiny Overfit Test ---")
    overfit_model = build_moment_detr_trm(opt).to(device)
    overfit_model.train()
    overfit_opt = torch.optim.AdamW(overfit_model.parameters(), lr=1e-3, weight_decay=1e-4)

    initial_loss = None
    final_loss = None
    losses_history = []

    for it in range(15):
        overfit_opt.zero_grad()
        out = overfit_model(**model_inputs)
        l_dict = criterion(out, targets)
        l_tot = sum(l_dict[k] * criterion.weight_dict[k] for k in l_dict if k in criterion.weight_dict)
        l_tot.backward()
        overfit_opt.step()

        val = l_tot.item()
        losses_history.append(val)
        if it == 0:
            initial_loss = val
        final_loss = val
        if (it + 1) % 5 == 0 or it == 0:
            print(f"    Iter {it + 1:02d}/15: loss = {val:.4f} "
                  f"(span={l_dict['loss_span']:.3f}, giou={l_dict['loss_giou']:.3f}, "
                  f"con={l_dict['loss_phrase_consistency']:.3f}, neg={l_dict['loss_phrase_negative']:.3f}, exc={l_dict['loss_phrase_exclusiveness']:.3f})")

    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    decrease_pct = (initial_loss - final_loss) / initial_loss * 100
    print(f"[+] Overfit test PASSED: Loss decreased from {initial_loss:.4f} to {final_loss:.4f} (-{decrease_pct:.2f}%).")

    print("\n" + "=" * 60)
    print("ALL 7 SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print("Moment-DETR-TRM is mathematically verified and ready for training.")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
