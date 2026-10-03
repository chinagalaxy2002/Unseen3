#!/usr/bin/env python
"""Fit per-split background PCA from train S+ clips outside GT windows only."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "moment_detr_gmr"))
from training.moment_detr_trm_gmr_joint_v2.train import build_dataset_config_joint
from training.moment_detr_trm_gmr_joint_v2.config import BaseOptionsJoint
from training.moment_detr_trm_gmr_joint_v2.dataset import StartEndDatasetJoint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("--max-samples",type=int,default=100000); ap.add_argument("--seed",type=int,default=3407); ap.add_argument("--output",default=None)
    a=ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed)
    mgr=BaseOptionsJoint("moment_detr_trm_gmr_joint_v2","charades_sta_semantic_novelty","clip_slowfast"); mgr.parse(); opt=mgr.option
    opt.train_path=str(ROOT/f"data/release/semantic_existence_v2/{a.split}/train.jsonl"); opt.max_v_l=200; opt.clip_length=1
    opt.v_feat_dirs=[str(ROOT/"features/charades_video/vid_slowfast"),str(ROOT/"features/charades_video/vid_clip")]
    opt.t_feat_dir=str(ROOT/"features/semantic_existence_v2/shared_clip_text"); opt.phrase_feat_dir=str(ROOT/"features/phrase_data/clip_phrase")
    ds=StartEndDatasetJoint(**build_dataset_config_joint(opt,opt.train_path,load_labels=True,keep_empty_gt=True,partition_filter=["S+"]))
    # Deterministic Algorithm R reservoir; only raw visual rows outside train S+ GT.
    reservoir=None; used=0; seen=0; D=None
    for i in range(len(ds)):
        ex=ds[i]; visual=ex["model_inputs"]["src_visual"].numpy(); D=visual.shape[1]; meta=ex["meta"]; T=len(visual)
        if reservoir is None: reservoir=np.empty((a.max_samples,D),dtype=np.float32)
        bg=np.ones(T,dtype=bool)
        for st,ed in meta.get("relevant_windows",[]):
            lo=max(0,min(T,int(float(st)/opt.clip_length))); hi=max(lo,min(T,int(float(ed)/opt.clip_length)))
            bg[lo:hi]=False
        for row in visual[bg]:
            seen+=1
            if used<a.max_samples: reservoir[used]=row; used+=1
            else:
                j=random.randrange(seen)
                if j<a.max_samples: reservoir[j]=row
    if not used: raise RuntimeError("No train S+ background visual vectors available")
    X=reservoir[:used]; K=min(256,D-1)
    try:
        from sklearn.decomposition import IncrementalPCA
        ipca=IncrementalPCA(n_components=K,batch_size=max(K,2048))
        # IncrementalPCA requires each partial batch >= K.
        batch=max(K,2048)
        for st in range(0,len(X),batch):
            chunk=X[st:st+batch]
            if len(chunk)>=K: ipca.partial_fit(chunk)
        mu=ipca.mean_.astype(np.float32); comps=ipca.components_.astype(np.float32)
    except ImportError:
        mu=X.mean(0,dtype=np.float64)
        cov=np.zeros((D,D),dtype=np.float64)
        for st in range(0,len(X),1024):
            xc=X[st:st+1024].astype(np.float64)-mu
            cov += xc.T @ xc
        _,vec=np.linalg.eigh(cov)
        comps=vec[:,-K:][:,::-1].T.astype(np.float32); mu=mu.astype(np.float32)
    residual=[]
    for st in range(0,len(X),1024):
        x=X[st:st+1024]-mu; res=x-(x@comps.T)@comps; residual.extend(np.linalg.norm(res,axis=1).tolist())
    output=Path(a.output or ROOT/f"results/moment_detr_trm_gmr_joint_v2/{a.split}/background_pca.npz"); output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,mu_bg=mu,components_bg=comps,residual_mean_bg=np.float32(np.mean(residual)),residual_std_bg=np.float32(max(np.std(residual),1e-6)),visual_feature_dim=D,pca_rank=K,background_sample_count=len(X),background_population_count=seen,seed=a.seed,source="train S+ background only",feature_order="SlowFast then CLIP visual; each modality row-wise L2 normalized; TEF excluded")
    print(json.dumps({"split":a.split,"path":str(output),"visual_feature_dim":D,"pca_rank":K,"background_sample_count":len(X),"background_population_count":seen},indent=2))
if __name__=="__main__": main()
