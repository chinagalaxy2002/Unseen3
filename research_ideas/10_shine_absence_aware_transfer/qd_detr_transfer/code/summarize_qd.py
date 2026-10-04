"""Write a descriptive A1/C1 macro only when both evaluations are complete."""
import json
import os
from pathlib import Path
import numpy as np
from train_qd_saliency import ROOT,digest
OUT=ROOT/'records/evaluation'

def main():
    rows=[]
    for split in ['A1','C1']:
        p=OUT/split/'status.json'
        if not p.exists():return
        try:state=json.loads(p.read_text())
        except json.JSONDecodeError:return
        if state['status']!='completed':return
        p=OUT/split/'RESULTS.json'
        if digest(p)!=state['results_sha256']:raise RuntimeError('Result mismatch')
        rows.append(json.loads(p.read_text()))
    macro={arm:{role:float(np.mean([r['models'][arm][role]['pooled_auroc'] for r in rows])) for role in ['seen','unseen']} for arm in ['canonical','B0','S1_saliency_only']}
    for arm,m in macro.items():m['gap']=m['seen']-m['unseen']
    b=macro['B0'];s=macro['S1_saliency_only'];delta={'seen':s['seen']-b['seen'],'unseen':s['unseen']-b['unseen'],'gap_reduction':b['gap']-s['gap']}
    result={'status':'completed','backbone':'QD-DETR-GMR','seed':3407,'epochs':50,'exploratory':True,'splits':rows,'macro':macro,'macro_S1_minus_B0':delta,'limitations':['single seed; intervals conditional on trained model','A1/C1 only, not five split','no independent-split macro confidence interval','mechanism Cq/Cv and closed quartets absent']}
    def atomic(path,text):
        tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(path)
    atomic(OUT/'RESULTS.json',json.dumps(result,indent=2)+'\n')
    lines=['# QD-DETR-GMR + SHINE coarse/fine: A1/C1 50 epochs','', '| Split | Arm | Seen | Unseen | Gap |','|---|---|---:|---:|---:|']
    for r in rows:
        for arm,m in r['models'].items():lines.append(f"| {r['split']} | {arm} | {m['seen']['pooled_auroc']:.4f} | {m['unseen']['pooled_auroc']:.4f} | {m['gap']:.4f} |")
    for arm,m in macro.items():lines.append(f"| Macro | {arm} | {m['seen']:.4f} | {m['unseen']:.4f} | {m['gap']:.4f} |")
    lines+=['',f"S1−B0 macro: ΔSeen {100*delta['seen']:+.2f}pp; ΔUnseen {100*delta['unseen']:+.2f}pp; gap reduction {100*delta['gap_reduction']:+.2f}pp.", '', 'Two-split single-seed exploratory comparison; gap reduction alone is not success. Per-split paired intervals and conditions in RESULTS.json.']
    atomic(OUT/'RESULTS.md','\n'.join(lines)+'\n')
if __name__=='__main__':main()
