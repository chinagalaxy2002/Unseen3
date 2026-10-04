"""Two-split descriptive summary for direct local evidence evaluation."""
import json
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'records/evaluation_v1'
rows=[]
for split in ['A1','C1']:
 status=json.loads((OUT/split/'status.json').read_text())
 if status['status']!='completed':raise RuntimeError('Evaluation incomplete')
 rows.append(json.loads((OUT/split/'RESULTS.json').read_text()))
macro={arm:{role:float(np.mean([r['models'][arm][role]['pooled_auroc'] for r in rows])) for role in ['seen','unseen']} for arm in rows[0]['models']}
for arm,m in macro.items():
 m['gap']=m['seen']-m['unseen']
 m['delta_seen_vs_B0']=m['seen']-macro['B0']['seen'];m['delta_unseen_vs_B0']=m['unseen']-macro['B0']['unseen']
 m['gap_reduction_vs_B0']=macro['B0']['seen']-macro['B0']['unseen']-m['gap']
 s=macro['S1_saliency_only'];m['delta_unseen_vs_S1']=m['unseen']-s['unseen'];m['delta_seen_vs_S1']=m['seen']-s['seen']
report={'status':'completed','backbone':'Moment-DETR-GMR','seed':3407,'epochs':10,'exploratory':True,'test_used_for_selection':False,'splits':rows,'macro':macro,'limitations':['A1/C1 already examined, no untouched confirmation','paired video CI conditional on one trained seed','descriptive macro, no split independence assumption','inference branch decomposition is not retraining ablation']}
(OUT/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Moment-DETR: direct local evidence, formal A1/C1 exploratory evaluation','',
 'All new arms10epochs, seed3407. Selected checkpoints and thresholds use Seen-val only. References are frozen v2 matched10epoch B0/S1.','',
 '| Arm | A1 Seen | A1 U | C1 Seen | C1 U | Macro ΔSeen vs B0(pp) | Macro ΔU vs B0(pp) | Macro ΔU vs S1(pp) | Macro Gap reduction vs B0(pp) |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for arm,m in macro.items():
 a,c=[r['models'][arm] for r in rows]
 lines.append(f"| {arm} | {a['seen']['pooled_auroc']:.4f} | {a['unseen']['pooled_auroc']:.4f} | {c['seen']['pooled_auroc']:.4f} | {c['unseen']['pooled_auroc']:.4f} | {100*m['delta_seen_vs_B0']:+.2f} | {100*m['delta_unseen_vs_B0']:+.2f} | {100*m['delta_unseen_vs_S1']:+.2f} | {100*m['gap_reduction_vs_B0']:+.2f} |")
lines+=['','Local_CF vs Uniform_CF controls capacity and direct encoder pooling; Local_CF vs Local_noCF tests CF supervision. Compare Seen retention, U and conditionals together. Per-split paired CIs, residual/base decomposition and shuffled-video diagnostics are in split RESULTS.json. No new training or coefficient selection is triggered by this evaluation.']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(macro,indent=2))
