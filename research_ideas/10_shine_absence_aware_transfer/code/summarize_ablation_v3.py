"""Report BCE/pair separation using hash-verified v2 controls, Seen-only."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NEW=['BCE_only','Pair_only','Weak_BCE_pair']

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);a=p.parse_args()
    f=json.loads((ROOT/'configs/ABLATION_V3_FREEZE.json').read_text())
    for name,h in f['reference_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise RuntimeError(f'Changed reference: {name}')
    results={}
    for version,arm in [('v2','B0'),('v2','S1_saliency_only'),('v2','SE1_full')]+[('v3',arm) for arm in NEW]:
        run=ROOT/f'runs/ablation_{version}'/a.fold/arm/f'seed3407/component_{version}_10ep'
        state=json.loads((run/'status.json').read_text());history=json.loads((run/'history.json').read_text())
        if state['status']!='completed' or state['epochs_done']!=10 or len(history)!=10:raise RuntimeError(f'Incomplete {run}')
        if hashlib.sha256((run/'best.ckpt').read_bytes()).hexdigest()!=state['checkpoint_sha256']:raise RuntimeError(f'Checkpoint mismatch {run}')
        results[arm]={'version':version,'metrics':json.loads((run/'metrics.json').read_text()),'history':history,'checkpoint_sha256':state['checkpoint_sha256']}
    exposures=[[h['exposure_sha256'] for h in row['history']] for row in results.values()]
    if not all(e==exposures[0] for e in exposures):raise RuntimeError('Exposure mismatch against v2 controls')
    baseline=results['B0']['metrics']['seen_val']['pooled_auroc'];s1=results['S1_saliency_only']['metrics']['seen_val']['pooled_auroc']
    lines=[f'# {a.fold} BCE/pair separation and weak-BCE repair','',
        'seed3407; 10 epochs; Seen-only; epoch0 excluded from selection. V2 controls are frozen historical references with verified exposure matches.','',
        '| Arm | Epoch1 | Epoch3 | Epoch10 | Best epoch / AUC | Δ vs B0 (pp) | Δ vs S1 (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,row in results.items():
        m=row['metrics'];h=row['history'];best=m['seen_val']['pooled_auroc'];vals=[h[i-1]['seen']['pooled_auroc'] for i in [1,3,10]]
        lines.append(f"| {arm} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {m['best_epoch']} / {best:.4f} | {100*(best-baseline):+.2f} | {100*(best-s1):+.2f} |")
    lines+=['','These are Seen development results, not evidence of Unseen gain. Inspect gradient geometry, GMR loss, conditionals and localization before selecting a further repair.']
    out=ROOT/'records/ablation_v3'
    (out/f'{a.fold}_RESULTS.json').write_text(json.dumps({'fold':a.fold,'formal_U_accessed':False,'all_exposure_hashes_match':True,'results':results},indent=2)+'\n')
    (out/f'{a.fold}_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
