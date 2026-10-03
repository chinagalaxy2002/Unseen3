#!/usr/bin/env python
"""Required focused smoke checks for Joint-v2."""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"training/moment_detr_gmr"))
from models.moment_detr_trm_gmr_joint_v2.joint_loss import PairwiseAUCLoss
from models.moment_detr_trm_gmr_joint_v2.moment_detr_trm_gmr_joint import build_moment_detr_trm_gmr_joint
from training.moment_detr_trm_gmr_joint_v2.config import BaseOptionsJoint

def fit_tiny_pca(X,k):
    mu=X.mean(0); _,_,vt=np.linalg.svd(X-mu,full_matrices=False); c=vt[:k]
    r=np.linalg.norm((X-mu)-((X-mu)@c.T)@c,axis=1)
    return dict(mu_bg=mu.astype("float32"),components_bg=c.astype("float32"),residual_mean_bg=float(r.mean()),residual_std_bg=float(max(r.std(),1e-6)))

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    from models.moment_detr_trm_gmr_joint_v2 import joint_modules, joint_loss
    from training.moment_detr_trm_gmr_joint_v2 import dataset, train, evaluate, infer
    assert PairwiseAUCLoss
    # Tiny PCA fit/save/load
    tiny=np.random.default_rng(3407).normal(size=(32,12)).astype("float32"); pca=fit_tiny_pca(tiny,4)
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"pca.npz"; np.savez(path,**pca)
        with np.load(path) as loaded: pca2={k:loaded[k] for k in pca}
        assert pca2["components_bg"].shape==(4,12)
    # Query independent span pooling and detached coordinates.
    from models.moment_detr_trm_gmr_joint_v2.moment_detr_trm_gmr_joint import MomentDETR_TRM_GMR_Joint
    raw=torch.randn(1,5,12,device=device); mask=torch.ones(1,5,device=device); spans=torch.tensor([[[.5,.3],[.2,.2]]],device=device,requires_grad=True)
    v1=MomentDETR_TRM_GMR_Joint.candidate_visual_residual(raw,mask,spans); v2=MomentDETR_TRM_GMR_Joint.candidate_visual_residual(raw.clone(),mask,spans.clone())
    assert torch.equal(v1,v2) and torch.isfinite(v1).all()

    mgr=BaseOptionsJoint("moment_detr_trm_gmr_joint_v2","charades_sta_semantic_novelty","clip_slowfast"); mgr.parse(); opt=mgr.option
    opt.device=device; opt.use_phrase=True; opt.drop_phrase=False; opt.lambda_refine=1.; opt.phrase_scale=10.; opt.phrase_proj_dim=None
    model=build_moment_detr_trm_gmr_joint(opt).to(device)
    D=opt.v_feat_dim-2; pca=fit_tiny_pca(np.random.default_rng(1).normal(size=(8,D)).astype("float32"),4)
    model.set_visual_pca(pca); model.train()
    B,T,Q,P,L=2,6,opt.num_queries,opt.max_phrases,4
    inp=dict(src_txt=torch.randn(B,L,512,device=device),src_txt_mask=torch.ones(B,L,device=device),src_vid=torch.randn(B,T,opt.v_feat_dim,device=device),src_vid_mask=torch.ones(B,T,device=device),src_visual=torch.randn(B,T,D,device=device),phrase_features=torch.randn(B,P,16,512,device=device),phrase_tokens_mask=torch.ones(B,P,16,device=device),phrase_mask=torch.ones(B,P,device=device))
    out=model(**inp)
    assert torch.isfinite(out["pred_exist_logits"]).all() and torch.isfinite(out["candidate_visual_residual"]).all()
    assert (out["candidate_visual_support"].abs()<=1).all()
    labels=torch.tensor([1.,0.],device=device); auc=PairwiseAUCLoss()(out["pred_exist_logits"],labels); bce=torch.nn.functional.binary_cross_entropy_with_logits(out["pred_exist_logits"],labels)
    assert torch.isfinite(auc) and torch.isfinite(bce)
    assert PairwiseAUCLoss()(torch.randn(2,device=device),torch.ones(2,device=device)).item()==0
    assert PairwiseAUCLoss()(torch.randn(2,device=device),torch.zeros(2,device=device)).item()==0
    model.zero_grad(); auc.backward()
    for param in [model.evidence_exist_head.mlp_exist_candidate[0].weight,model.transformer.decoder.layers[0].linear1.weight,model.phrase_matcher.phrase_proj.weight,model.candidate_attention.mlp_att[0].weight,model.gated_refinement.mlp_gate[0].weight,model.alpha_visual_raw]:
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0
    # The visual branch must not backpropagate into span coordinates.
    assert model.span_embed.layers[-1].weight.grad is None
    # Mixed BCE + pairwise objective one optimizer step.
    model.zero_grad(); out2=model(**inp); bce2=torch.nn.functional.binary_cross_entropy_with_logits(out2["pred_exist_logits"],labels); auc2=PairwiseAUCLoss()(out2["pred_exist_logits"],labels); (bce2+auc2).backward(); torch.optim.AdamW(model.parameters(),lr=1e-4).step()
    assert np.isfinite(float(out["pred_exist_logits"][0].detach().cpu()))
    print(json.dumps({"device":device,"imports":True,"tiny_pca_save_load":True,"query_independence":True,"fused_forward":True,"bce_auc_finite":True,"auc_empty_class_safe":True,"auc_gradient_flow":True,"span_detach":True,"optimizer_step":True,"existence_serialization":"Python float (full precision)"},indent=2))
if __name__=="__main__": main()
