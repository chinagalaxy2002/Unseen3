#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/trm_gmr_joint_v2'
RES=ROOT/'results/moment_detr_trm_gmr_joint_v2'
SPLITS=['A1','A2_alt','A3','C1','C2_alt']
rows=[]
for split in SPLITS:
    d=json.loads((RES/split/'joint_v2_summary.json').read_text())
    with np.load(RES/split/'background_pca.npz') as pca:
        d['pca_rank']=int(pca['pca_rank'])
        d['background_sample_count']=int(pca['background_sample_count'])
    train=json.loads((RES/split/'training_meta.json').read_text())
    d['auc_valid_batch_fraction']=train.get('auc_valid_batch_fraction')
    d['pca_source']='train S+ background only'
    rows.append(d)
keys=['split','joint_v1_unseen_auroc','unseen_semantic_auroc','unseen_fused_auroc','delta_v2_vs_v1_unseen_auroc','delta_unseen_visual_fusion','seen_fused_auroc','auroc_gap','U+_FRR','U-_RR','matched_pair_acc','U+_raw_R1@0.5']
table='| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'
for d in rows:
    vals=[]
    for k in keys:
        x=d.get(k)
        vals.append(x if isinstance(x,str) else ('' if x is None else f'{x:.6f}'))
    table+='| '+' | '.join(vals)+' |\n'
def avg(k): return float(np.mean([d[k] for d in rows if d.get(k) is not None]))
improved=sum(d['delta_v2_vs_v1_unseen_auroc']>0 for d in rows)
degraded=sum(d['delta_v2_vs_v1_unseen_auroc']<0 for d in rows)
summary={'splits':rows,'macro_seen_auroc':avg('seen_fused_auroc'),'macro_unseen_auroc':avg('unseen_fused_auroc'),'macro_seen_unseen_gap':avg('auroc_gap'),'macro_delta_v2_vs_v1':avg('delta_v2_vs_v1_unseen_auroc'),'improved_split_count':improved,'degraded_split_count':degraded}
(OUT/'multi_split_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
md='# Joint-v2 five-split exploratory results\n\nJoint-v2 is an exploratory follow-up motivated by Joint-v1 failure modes. These U results are not an untouched unseen evaluation. U was excluded from this run\'s PCA fitting, training, selection, and calibration.\n\n'+table+f"\nMacro Seen AUROC: {summary['macro_seen_auroc']:.6f}\n\nMacro Unseen AUROC: {summary['macro_unseen_auroc']:.6f}\n\nMacro Seen-Unseen gap: {summary['macro_seen_unseen_gap']:.6f}\n\nMacro ΔV2-vs-V1 unseen AUROC: {summary['macro_delta_v2_vs_v1']:.6f}\n\nImproved splits: {improved}/5; degraded splits: {degraded}/5.\n"
(OUT/'MULTI_SPLIT_RESULT.md').write_text(md)
print(md)
