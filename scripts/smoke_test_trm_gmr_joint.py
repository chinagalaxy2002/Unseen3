"""
Comprehensive Smoke Test for Moment-DETR-TRM-GMR-Joint-v1.
Verifies end-to-end integration, gradient flow, loss finiteness,
and dataset integrity across S+ and S- queries.
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
from easydict import EasyDict


def run_smoke_test():
    print("=" * 70)
    print("STARTING MOMENT-DETR-TRM-GMR-JOINT SMOKE TEST SUITE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # ========================================================
    # STEP A: Python Import Test
    # ========================================================
    print("\n--- [Step A] Python Import Test ---")
    try:
        from models.moment_detr_trm_gmr_joint.joint_modules import (
            PhraseSlotMatcher,
            CandidateConditionedPhraseAttention,
            CandidateWiseGatedRefinement,
            EvidenceAwareExistenceHead,
        )
        from models.moment_detr_trm_gmr_joint.moment_detr_trm_gmr_joint import (
            MomentDETR_TRM_GMR_Joint,
            build_moment_detr_trm_gmr_joint,
        )
        from models.moment_detr_trm_gmr_joint.joint_loss import (
            SetCriterionJoint,
            build_criterion_joint,
        )
        from training.moment_detr_trm_gmr_joint.dataset import (
            StartEndDatasetJoint,
            start_end_collate_joint,
            prepare_batch_inputs_joint,
        )
        from training.moment_detr_trm_gmr_joint.config import BaseOptionsJoint
        from training.moment_detr_trm_gmr_joint.evaluate import eval_epoch_joint
        from training.moment_detr_trm_gmr_joint.train import (
            setup_model_and_criterion,
            build_dataset_config_joint,
        )
        from training.moment_detr_trm_gmr_joint.infer import run_full_inference
        print("[+] Successfully imported all joint models and training/eval/infer modules.")
    except Exception as e:
        print(f"[!] FAILED Step A Import Test: {e}")
        raise e

    # ========================================================
    # STEP B: Dataset Smoke Test (S+ and S- queries)
    # ========================================================
    print("\n--- [Step B] Dataset Smoke Test (S+ and S-) ---")
    opt_mgr = BaseOptionsJoint(
        "moment_detr_trm_gmr_joint",
        "charades_sta_semantic_novelty",
        "clip_slowfast",
    )
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

    dset_config = build_dataset_config_joint(
        opt,
        opt.train_path,
        load_labels=True,
        keep_empty_gt=True,
        partition_filter=["S+", "S-"],
    )
    dataset = StartEndDatasetJoint(**dset_config)
    print(f"[+] Dataset initialized successfully. Total training samples (S+ and S-): {len(dataset)}")

    # Find S+ and S- samples
    s_pos_samples = []
    s_neg_samples = []
    for i in range(len(dataset)):
        s = dataset[i]
        part = s["meta"].get("partition", "")
        if part == "S+" and len(s_pos_samples) < 2:
            s_pos_samples.append(s)
        elif part == "S-" and len(s_neg_samples) < 2:
            s_neg_samples.append(s)
        if len(s_pos_samples) >= 2 and len(s_neg_samples) >= 2:
            break

    print(f"[+] Found {len(s_pos_samples)} S+ samples and {len(s_neg_samples)} S- samples.")
    for i, s in enumerate(s_pos_samples + s_neg_samples):
        meta = s["meta"]
        m_in = s["model_inputs"]
        print(f"    Sample {i} [{meta.get('partition')}]: qid={meta['qid']}, "
              f"GT_exist={meta.get('existence_label')}, windows={meta.get('relevant_windows')}")
        assert "phrase_features" in m_in
        assert "phrase_tokens_mask" in m_in
        assert "phrase_mask" in m_in
        assert m_in["phrase_features"].shape == (10, 16, 512)

    # Test Collate with mixed batch
    mixed_samples = s_pos_samples + s_neg_samples
    batch_meta, batched_inputs = start_end_collate_joint(mixed_samples)
    model_inputs, targets = prepare_batch_inputs_joint(batched_inputs, device)
    B = len(mixed_samples)
    assert model_inputs["src_vid"].shape[0] == B
    assert targets["exist_label"].shape == (B,)
    print(f"[+] Mixed batch collated successfully with B={B}.")
    print(f"    exist_label: {targets['exist_label'].tolist()}")

    # ========================================================
    # STEP C: Model Forward Test
    # ========================================================
    print("\n--- [Step C] Model Forward Test ---")
    model = build_moment_detr_trm_gmr_joint(opt).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**model_inputs)

    required_keys = [
        "pred_spans", "pred_logits_base", "pred_logits",
        "pred_exist_logits", "pred_phrase_scores", "raw_cosine",
        "candidate_phrase_support", "candidate_phrase_attention",
        "candidate_phrase_gate", "candidate_exist_evidence",
        "base_margin", "refined_margin",
        "phrase_mask", "phrase_repr", "slot_repr", "p_proj", "s_proj",
    ]
    for k in required_keys:
        assert k in outputs, f"Missing output key: {k}"
        assert outputs[k] is not None, f"Output key {k} is None"

    Q = opt.num_queries
    P = opt.max_phrases
    print(f"[+] Output shapes verified:")
    print(f"    pred_spans: {outputs['pred_spans'].shape} (expected [{B}, {Q}, 2])")
    print(f"    pred_logits: {outputs['pred_logits'].shape} (expected [{B}, {Q}, 2])")
    print(f"    pred_exist_logits: {outputs['pred_exist_logits'].shape} (expected [{B}])")
    print(f"    candidate_phrase_attention: {outputs['candidate_phrase_attention'].shape} (expected [{B}, {P}, {Q}])")
    print(f"    candidate_phrase_support: {outputs['candidate_phrase_support'].shape} (expected [{B}, {Q}])")
    print(f"    candidate_phrase_gate: {outputs['candidate_phrase_gate'].shape} (expected [{B}, {Q}])")
    print(f"    candidate_exist_evidence: {outputs['candidate_exist_evidence'].shape} (expected [{B}, {Q}])")

    # Value bound assertions
    assert (outputs["pred_spans"] >= 0.0).all() and (outputs["pred_spans"] <= 1.0).all(), "pred_spans outside [0, 1]"
    assert (outputs["candidate_phrase_gate"] >= 0.0).all() and (outputs["candidate_phrase_gate"] <= 1.0).all(), "candidate_phrase_gate outside [0, 1]"
    assert (outputs["candidate_phrase_support"] >= -1.01).all() and (outputs["candidate_phrase_support"] <= 1.01).all(), "candidate_phrase_support outside [-1, 1]"

    # Att weights sum to 1 over phrases
    att_sum = outputs["candidate_phrase_attention"].sum(dim=1)
    assert torch.allclose(att_sum, torch.ones_like(att_sum), atol=1e-4), "Candidate attention weights do not sum to 1"
    print("[+] All forward shape and range assertions PASSED.")

    # ========================================================
    # STEP D: Loss Test & S- / Empty GT Safety
    # ========================================================
    print("\n--- [Step D] Loss Test & S- / Empty GT Safety ---")
    criterion = build_criterion_joint(opt).to(device)
    loss_dict = criterion(outputs, targets)

    print("[+] Computed joint loss dictionary on mixed batch:")
    for k, v in loss_dict.items():
        val = float(v)
        print(f"    {k}: {val:.4f} (weight={criterion.weight_dict.get(k, 0.0)})")
        assert not torch.isnan(v), f"Loss {k} is NaN"
        assert not torch.isinf(v), f"Loss {k} is Inf"

    expected_loss_terms = [
        "loss_span", "loss_giou", "loss_label",
        "loss_phrase_consistency", "loss_phrase_negative", "loss_phrase_exclusiveness",
        "loss_exist",
    ]
    for term in expected_loss_terms:
        assert term in loss_dict, f"Missing loss term: {term}"

    # Pure S- batch loss test
    s_neg_meta, s_neg_batched = start_end_collate_joint(s_neg_samples)
    s_neg_inputs, s_neg_targets = prepare_batch_inputs_joint(s_neg_batched, device)
    with torch.no_grad():
        s_neg_outputs = model(**s_neg_inputs)
    neg_loss_dict = criterion(s_neg_outputs, s_neg_targets)
    for k, v in neg_loss_dict.items():
        assert not torch.isnan(v), f"S- loss {k} is NaN"
        assert not torch.isinf(v), f"S- loss {k} is Inf"
    print("[+] Pure S- batch loss test PASSED: all losses finite and stable.")

    # ========================================================
    # STEP E: End-to-End Gradient Flow Test
    # ========================================================
    print("\n--- [Step E] End-to-End Gradient Flow Test ---")
    model.train()
    criterion.train()

    # Sub-test E.1: Total loss backward
    train_outputs = model(**model_inputs)
    train_loss_dict = criterion(train_outputs, targets)
    total_loss = sum(
        train_loss_dict[k] * criterion.weight_dict[k]
        for k in train_loss_dict.keys()
        if k in criterion.weight_dict
    )
    model.zero_grad()
    total_loss.backward()

    checks = [
        ("model.phrase_matcher.phrase_proj.weight", model.phrase_matcher.phrase_proj.weight.grad),
        ("model.phrase_matcher.slot_proj.weight", model.phrase_matcher.slot_proj.weight.grad),
        ("model.candidate_attention.mlp_att.0.weight", model.candidate_attention.mlp_att[0].weight.grad),
        ("model.gated_refinement.mlp_gate.0.weight", model.gated_refinement.mlp_gate[0].weight.grad),
        ("model.evidence_exist_head.mlp_exist_candidate.0.weight", model.evidence_exist_head.mlp_exist_candidate[0].weight.grad),
        ("model.input_txt_proj", list(model.input_txt_proj.parameters())[0].grad),
        ("model.input_vid_proj", list(model.input_vid_proj.parameters())[0].grad),
        ("model.transformer.encoder.layers[0].linear1.weight", model.transformer.encoder.layers[0].linear1.weight.grad),
        ("model.transformer.decoder.layers[0].linear1.weight", model.transformer.decoder.layers[0].linear1.weight.grad),
    ]

    for name, grad in checks:
        assert grad is not None, f"Gradient is None for {name}"
        norm = grad.norm().item()
        assert norm > 0, f"Gradient is zero for {name}"
        assert not torch.isnan(grad).any(), f"Gradient is NaN for {name}"
        print(f"    [Total Loss] {name}: grad_norm = {norm:.6f}")

    print("[+] Total loss backprop verified on all joint modules and transformer backbone.")

    # Sub-test E.2: CRUCIAL CHECK - Backward on L_exist ALONE
    print("\n--- [Sub-test E.2] Backward on L_exist ALONE ---")
    model.zero_grad()
    train_outputs_2 = model(**model_inputs)
    train_loss_dict_2 = criterion(train_outputs_2, targets)
    exist_loss_only = train_loss_dict_2["loss_exist"]
    exist_loss_only.backward()

    exist_gradient_checks = [
        ("model.evidence_exist_head.mlp_exist_candidate.0.weight", model.evidence_exist_head.mlp_exist_candidate[0].weight.grad),
        ("model.gated_refinement.mlp_gate.0.weight", model.gated_refinement.mlp_gate[0].weight.grad),
        ("model.candidate_attention.mlp_att.0.weight", model.candidate_attention.mlp_att[0].weight.grad),
        ("model.phrase_matcher.phrase_proj.weight", model.phrase_matcher.phrase_proj.weight.grad),
        ("model.phrase_matcher.slot_proj.weight", model.phrase_matcher.slot_proj.weight.grad),
        ("model.transformer.decoder.layers[0].linear1.weight", model.transformer.decoder.layers[0].linear1.weight.grad),
        ("model.input_txt_proj", list(model.input_txt_proj.parameters())[0].grad),
    ]

    for name, grad in exist_gradient_checks:
        assert grad is not None, f"L_exist gradient is None for {name}"
        norm = grad.norm().item()
        assert norm > 0, f"L_exist gradient is 0 for {name}"
        assert not torch.isnan(grad).any(), f"L_exist gradient is NaN for {name}"
        print(f"    [L_exist only] {name}: grad_norm = {norm:.6f}")

    print("[+] CRUCIAL VERIFICATION PASSED: L_exist successfully flows gradients through "
          "evidence head -> gated refinement -> candidate attention -> phrase matcher -> transformer!")

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
    overfit_model = build_moment_detr_trm_gmr_joint(opt).to(device)
    overfit_model.train()
    overfit_opt = torch.optim.AdamW(overfit_model.parameters(), lr=1e-3, weight_decay=1e-4)

    initial_loss = None
    final_loss = None

    for it in range(15):
        overfit_opt.zero_grad()
        out = overfit_model(**model_inputs)
        l_dict = criterion(out, targets)
        l_tot = sum(l_dict[k] * criterion.weight_dict[k] for k in l_dict if k in criterion.weight_dict)
        l_tot.backward()
        overfit_opt.step()

        val = l_tot.item()
        if it == 0:
            initial_loss = val
        final_loss = val
        if (it + 1) % 5 == 0 or it == 0:
            print(f"    Iter {it + 1:02d}/15: total_loss = {val:.4f} "
                  f"(span={l_dict['loss_span']:.3f}, giou={l_dict['loss_giou']:.3f}, "
                  f"con={l_dict['loss_phrase_consistency']:.3f}, exist={l_dict['loss_exist']:.3f})")

    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    decrease_pct = (initial_loss - final_loss) / initial_loss * 100
    print(f"[+] Overfit test PASSED: Loss decreased from {initial_loss:.4f} to {final_loss:.4f} (-{decrease_pct:.2f}%).")

    print("\n" + "=" * 70)
    print("ALL 7 SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print("Moment-DETR-TRM-GMR-Joint-v1 is mathematically verified and ready for formal training.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
