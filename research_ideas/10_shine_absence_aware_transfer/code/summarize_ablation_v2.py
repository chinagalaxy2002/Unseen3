"""Seen-only descriptive ablation report; does not choose repair hyperparameters."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=['B0','S1_saliency_only','E1_rotated_exist_only','SE1_full']

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);a=p.parse_args()
    results={};lines=[f'# {a.fold} component ablation: Seen-only development','',
        'seed3407; 10 epochs; matched auxiliary forwards; no formal test read. Epoch0 is a reference, excluded from checkpoint selection.','',
        '| Arm | Epoch0 | Epoch1 | Epoch3 | Epoch10 | Best epoch / AUC | Δbest vs B0 (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        run=ROOT/'runs/ablation_v2'/a.fold/arm/'seed3407/component_v2_10ep'
        state=json.loads((run/'status.json').read_text())
        if state['status']!='completed' or state['epochs_done']!=10: raise RuntimeError(f'Incomplete {run}')
        m=json.loads((run/'metrics.json').read_text());h=json.loads((run/'history.json').read_text())
        results[arm]={'metrics':m,'history':h,'checkpoint_sha256':state['checkpoint_sha256']}
    baseline=results['B0']['metrics']['seen_val']['pooled_auroc']
    for arm,record in results.items():
        m=record['metrics'];h=record['history'];vals=[h[i-1]['seen']['pooled_auroc'] for i in [1,3,10]]
        best=m['seen_val']['pooled_auroc'];delta=100*(best-baseline)
        lines.append(f"| {arm} | {m['epoch0']['seen']['pooled_auroc']:.4f} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {m['best_epoch']} / {best:.4f} | {delta:+.2f} |")
    lines.extend(['','These results diagnose Seen damage; they do not establish Unseen improvement. Gradient files describe local pre-clipping gradients.'])
    out=ROOT/'records/ablation_v2'
    (out/f'{a.fold}_RESULTS.json').write_text(json.dumps({'fold':a.fold,'formal_U_accessed':False,'results':results},indent=2)+'\n')
    (out/f'{a.fold}_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
