"""Report matched 50-epoch Seen-to-Unseen gap; no independent macro CI."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'records/saliency_v5/evaluation'
rows=[]
for split in ['A1','C1']:
    state=json.loads((OUT/split/'status.json').read_text())
    if state['status']!='completed':raise RuntimeError('Incomplete evaluation')
    rows.append(json.loads((OUT/split/'RESULTS.json').read_text()))
macro={arm:{role:float(np.mean([r['models'][arm][role]['pooled_auroc'] for r in rows])) for role in ['seen','unseen']} for arm in ['B0','S1_saliency_only']}
for m in macro.values():m['gap']=m['seen']-m['unseen']
b=macro['B0'];s=macro['S1_saliency_only']
comparison={'delta_seen':s['seen']-b['seen'],'delta_unseen':s['unseen']-b['unseen'],'gap_reduction':b['gap']-s['gap'],
    'relative_gap_reduction':(b['gap']-s['gap'])/b['gap']}
report={'status':'completed','epochs':50,'seed':3407,'scope':'A1/C1 exploratory matched-forward saliency-only comparison','splits':rows,'macro':macro,'comparison':comparison,'test_used_for_selection':False}
(OUT/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Coarse/fine-only vs baseline, 50 epochs','',
    'Fresh canonical initialization for each arm; seed3407; matched forwards and exposure; best trained epoch by Seen-val only. A1/C1 exploratory test.','',
    '| Split | B0 Seen / U | S1 Seen / U | B0 Gap (pp) | S1 Gap (pp) | Gap reduction (pp) | ΔU (pp) | ΔSeen (pp) |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    bb=row['models']['B0'];ss=row['models']['S1_saliency_only']
    lines.append(f"| {row['split']} | {bb['seen']['pooled_auroc']:.4f} / {bb['unseen']['pooled_auroc']:.4f} | {ss['seen']['pooled_auroc']:.4f} / {ss['unseen']['pooled_auroc']:.4f} | {100*bb['gap']:.2f} | {100*ss['gap']:.2f} | {100*(bb['gap']-ss['gap']):+.2f} | {100*(ss['unseen']['pooled_auroc']-bb['unseen']['pooled_auroc']):+.2f} | {100*(ss['seen']['pooled_auroc']-bb['seen']['pooled_auroc']):+.2f} |")
lines.append(f"| Macro | {b['seen']:.4f} / {b['unseen']:.4f} | {s['seen']:.4f} / {s['unseen']:.4f} | {100*b['gap']:.2f} | {100*s['gap']:.2f} | {100*comparison['gap_reduction']:+.2f} | {100*comparison['delta_unseen']:+.2f} | {100*comparison['delta_seen']:+.2f} |")
lines+=['','Positive gap reduction means a smaller Seen−Unseen gap; interpret jointly with ΔU and Seen retention. Per-split paired video-cluster CI/conditionals/localization/shuffle diagnostics are in split RESULTS.json. Macro describes two benchmark splits and has no independence-based CI.']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(comparison),flush=True)
