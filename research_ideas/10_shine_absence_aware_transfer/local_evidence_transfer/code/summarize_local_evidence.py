"""Seen-only matched-exposure ablation report; branch diagnostics included."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA_ROOT=ROOT.parent
NEW=['Local_CF','Uniform_CF','Local_noCF']
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);a=p.parse_args();results={}
 freeze=json.loads((ROOT/'configs/EXPERIMENT_FREEZE.json').read_text())
 for name,h in freeze['reference_sha256'].items():
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise RuntimeError(f'Changed reference {name}')
 for arm in ['B0','S1_saliency_only']+NEW:
  run=(DATA_ROOT/f'runs/ablation_v2/{a.fold}/{arm}/seed3407/component_v2_10ep') if arm not in NEW else (ROOT/f'runs/v1/{a.fold}/{arm}/seed3407/local_evidence_v1_10ep')
  state=json.loads((run/'status.json').read_text());history=json.loads((run/'history.json').read_text())
  if state['status']!='completed' or state['epochs_done']!=10 or len(history)!=10:raise RuntimeError('Incomplete '+str(run))
  if hashlib.sha256((run/'best.ckpt').read_bytes()).hexdigest()!=state['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
  result={'history':history,'metrics':json.loads((run/'metrics.json').read_text()),'checkpoint_sha256':state['checkpoint_sha256']}
  if arm in NEW:
   preds=[json.loads(line) for line in (run/'predictions_seen.jsonl').read_text().splitlines()]
   result['branch_diagnostics']={'mean_abs_local_logit':float(np.mean([abs(p['local_logit']) for p in preds])),
    'base_only_scores':'saved in predictions_seen.jsonl; trained decoder, not untouched canonical',
    'mean_pool_entropy':float(np.mean([p['pool_entropy'] for p in preds]))}
  results[arm]=result
 exposure=[[h['exposure_sha256'] for h in result['history']] for result in results.values()]
 if not all(e==exposure[0] for e in exposure):raise RuntimeError('Exposure mismatch against frozen controls')
 b=results['B0']['metrics']['seen_val']['pooled_auroc'];s=results['S1_saliency_only']['metrics']['seen_val']['pooled_auroc']
 lines=[f'# {a.fold}: local evidence to existence, Seen-only','',
 '10epochs, seed3407; old controls from v2 with verified input/exposure. No formal test access.','',
 '| Arm | Epoch1 | Epoch3 | Epoch10 | Best epoch / Seen AUC | Δ vs B0(pp) | Δ vs S1(pp) |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for arm,result in results.items():
  h=result['history'];m=result['metrics'];best=m['seen_val']['pooled_auroc'];vals=[h[i-1]['seen']['pooled_auroc'] for i in [1,3,10]]
  lines.append(f"| {arm} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {m['best_epoch']} / {best:.4f} | {100*(best-b):+.2f} | {100*(best-s):+.2f} |")
 lines+=['','Local_CF vs Uniform_CF controls branch capacity/global encoder pooling; Local_CF vs Local_noCF controls saliency supervision. Local_CF vs old S1 measures adding the direct branch. These Seen results alone do not establish Unseen gain.']
 out=ROOT/'records';(out/f'{a.fold}_RESULTS.json').write_text(json.dumps({'fold':a.fold,'formal_U_accessed':False,'all_epoch_exposures_match':True,'results':results},indent=2)+'\n');(out/f'{a.fold}_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
