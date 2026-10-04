"""Two-split macro is descriptive; do not assume independent splits for CI."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'records/ablation_test_v2_v3'
rows=[]
for split in ['A1','C1']:
 state=json.loads((OUT/split/'status.json').read_text())
 if state['status']!='completed':raise RuntimeError('Incomplete evaluation')
 rows.append(json.loads((OUT/split/'RESULTS.json').read_text()))
macro={arm:{role:float(np.mean([r['models'][arm][role]['pooled_auroc'] for r in rows])) for role in ['seen','unseen']} for arm in rows[0]['models']}
for arm,m in macro.items():
 m['gap']=m['seen']-m['unseen'];m['delta_seen_vs_B0']=m['seen']-macro['B0']['seen'];m['delta_unseen_vs_B0']=m['unseen']-macro['B0']['unseen']
report={'status':'completed','splits':rows,'macro':macro,'scope':'A1/C1 exploratory formal evaluation; seven 10-epoch arms plus canonical','test_used_for_selection':False,'next_stage':'v4 reduced-supervision design fixed in ABLATION_V4_PLAN before this evaluation','limitations':['single trained seed; CI conditional on trained models','A1/C1 test previously examined, remains exploratory','no independent-split macro CI','no Cq/Cv retraining or closed-quartet mechanism proof']}
(OUT/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# A1/C1 v2/v3 exploratory formal Seen/Unseen evaluation','',
       'All adapted arms: 10 epochs, seed3407; checkpoint selection and thresholds use Seen-val only. Formal test is exploratory. Canonical is an initialization reference, not a 10-epoch adapted arm.','',
       '| Arm | A1 Seen | A1 U | C1 Seen | C1 U | Macro ΔSeen vs B0 (pp) | Macro ΔU vs B0 (pp) |',
       '|---|---:|---:|---:|---:|---:|---:|']
for arm,m in macro.items():
 a,c=[r['models'][arm] for r in rows]
 lines.append(f"| {arm} | {a['seen']['pooled_auroc']:.4f} | {a['unseen']['pooled_auroc']:.4f} | {c['seen']['pooled_auroc']:.4f} | {c['unseen']['pooled_auroc']:.4f} | {m['delta_seen_vs_B0']*100:+.2f} | {m['delta_unseen_vs_B0']*100:+.2f} |")
lines+=['','Per-split intervals, conditions, localization and fresh shuffled-video diagnostics are in A1/RESULTS.json and C1/RESULTS.json. Macro is descriptive; splits share benchmark/video sources. Gap reduction alone does not establish success. V4 weights were fixed from Seen results before this evaluation and will not be adjusted using these test outcomes.']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(macro,indent=2))
