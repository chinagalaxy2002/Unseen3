#!/usr/bin/env python
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/trm_gmr_joint_v3';RES=ROOT/'results/moment_detr_trm_gmr_joint_v3'
SPLITS=['A1','A2_alt','A3','C1','C2_alt'];rows=[]
for split in SPLITS:
 p=RES/split/'joint_v3_summary.json'
 rows.append(json.loads(p.read_text()) if p.exists() else {'split':split,'status':'not_completed_or_job_failed'})
completed=[r for r in rows if r['status']=='completed']
def avg(k):return float(np.mean([r[k] for r in completed])) if completed else None
summary={'splits':rows,'completed_split_count':len(completed),'failed_or_pending_split_count':5-len(completed),
 'macro_seen_auroc':avg('seen_auroc'),'macro_unseen_auroc':avg('unseen_auroc'),'macro_seen_unseen_gap':avg('seen_unseen_gap'),
 'macro_delta_v3_vs_v1':avg('delta_v3_vs_v1_unseen_auroc'),'improved_split_count':sum(r['delta_v3_vs_v1_unseen_auroc']>0 for r in completed),
 'degraded_split_count':sum(r['delta_v3_vs_v1_unseen_auroc']<0 for r in completed),'macro_scope':'completed eligible splits only; report completion count alongside macro'}
keys=['split','status','best_epoch','epochs_trained','threshold','joint_v1_unseen_auroc','unseen_auroc','delta_v3_vs_v1_unseen_auroc','seen_auroc','seen_unseen_gap','U+_FRR','U-_RR','matched_pair_acc','U+_raw_R1@0.5','S+_raw_R1@0.5','best_seen_worst_semantic_auroc','best_seen_val_mAP','localization_floor_mAP']
md='# Joint-v3 exploratory results\n\nExploratory follow-up motivated by inspected Joint-v1/v2 U results. U was excluded from training, grouping decisions, checkpoint selection, reference localization floors, and calibration.\n\n'
md+='| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'
for r in rows:
 def fmt(x):return '—' if x is None else (f'{x:.6f}' if isinstance(x,float) else str(x))
 md+='| '+' | '.join(fmt(r.get(k)) for k in keys)+' |\n'
md+=f"\nCompleted eligible splits: {len(completed)}/5. Improved: {summary['improved_split_count']}; degraded: {summary['degraded_split_count']}.\n\nMacros cover completed eligible splits only.\n\n"
for k in ['macro_seen_auroc','macro_unseen_auroc','macro_seen_unseen_gap','macro_delta_v3_vs_v1']:md+=f'{k}: {summary[k]}\n\n'
(OUT/'multi_split_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(OUT/'MULTI_SPLIT_RESULT.md').write_text(md);print(md)
