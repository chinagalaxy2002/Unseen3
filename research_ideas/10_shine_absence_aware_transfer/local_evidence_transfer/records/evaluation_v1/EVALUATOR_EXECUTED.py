"""Selected Moment local-evidence checkpoints; exploratory formal test only."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_local_evidence import ROOT,DATA_ROOT,CachedDataset,build_model,evaluate,read,write,digest,ARM_POOLING
from evidence_bridge import attach_local_evidence
from evaluation_metrics import subset_diagnostic,shuffled_auc,interval
OUT=ROOT/'records/evaluation_v1'
NEW=['Local_CF','Uniform_CF','Local_noCF']
BASE=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2')

def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);p.add_argument('--device',required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 frozen=json.loads((OUT/'EVALUATION_FREEZE.json').read_text())
 for name,h in frozen['file_sha256'].items():
  if digest(name)!=h:raise RuntimeError(f'Changed frozen input: {name}')
 identities=frozen['checkpoints'][a.fold];dest=OUT/a.fold;dest.mkdir(exist_ok=False)
 write(dest/'status.json',{'status':'running','test_used_for_selection':False})
 try:
  canonical=torch.load(BASE/a.fold/'moment/best.ckpt',map_location='cpu',weights_only=False);opt=copy.deepcopy(canonical['opt']);opt.device=a.device
  ds=CachedDataset(opt,a.fold,'test');val=CachedDataset(opt,a.fold,'seen_val')
  row={'split':a.fold,'backbone':'Moment-DETR-GMR','epochs':10,'seed':3407,'exploratory':True,'test_used_for_selection':False,'test_sha256':digest(DATA_ROOT/f'data/formal/{a.fold}/test.jsonl'),'checkpoints':identities,'models':{},'comparisons':{}}
  subsets={}
  source=json.loads((DATA_ROOT/f'records/ablation_test_v2_v3/{a.fold}/RESULTS.json').read_text())
  for arm in ['B0','S1_saliency_only']:
   identity=identities[arm];pred_path=dest/(arm+'_test_predictions.jsonl');shutil.copyfile(identity['predictions_path'],pred_path)
   if digest(pred_path)!=identity['predictions_sha256']:raise RuntimeError('Reference predictions changed')
   row['models'][arm]=copy.deepcopy(source['models'][arm]);preds=read(pred_path)
   if [(str(r['qid']),r['vid'],r['exist_label']) for r in ds.rows]!=[(str(r['qid']),r['vid'],r['exist_label']) for r in preds]:raise RuntimeError('Reference row identities differ')
   threshold=row['models'][arm]['seen_val']['threshold_logit']
   seen,_,sp=subset_diagnostic(ds.rows,preds,'S',threshold);unseen,_,up=subset_diagnostic(ds.rows,preds,'U',threshold)
   for role,recomputed in [('seen',seen),('unseen',unseen)]:
    if abs(recomputed['pooled_auroc']-row['models'][arm][role]['pooled_auroc'])>1e-12:raise RuntimeError('Reference metric mismatch')
   subsets[arm]={'seen':sp,'unseen':up}
  for arm in NEW:
   saved=torch.load(identities[arm]['path'],map_location='cpu',weights_only=False)
   model,_=build_model(opt);model.load_state_dict(canonical['model'],strict=True);model=attach_local_evidence(model,ARM_POOLING[arm]);model.load_state_dict(saved['model'],strict=True);model.to(a.device).eval()
   validation,_=evaluate(model,val,opt)
   expected=json.loads(Path(identities[arm]['path']).with_name('metrics.json').read_text())['seen_val']['pooled_auroc']
   if abs(validation['pooled_auroc']-expected)>1e-12:raise RuntimeError('Selected Seen-val replay mismatch')
   threshold=validation['threshold_logit'];pred_path=dest/(arm+'_test_predictions.jsonl');evaluate(model,ds,opt,threshold=threshold,save=pred_path);preds=read(pred_path)
   seen,_,sp=subset_diagnostic(ds.rows,preds,'S',threshold);unseen,_,up=subset_diagnostic(ds.rows,preds,'U',threshold)
   branch={}
   for prefix,role in [('S','seen'),('U','unseen')]:
    selected=[i for i,r in enumerate(ds.rows) if r['partition'].startswith(prefix)];labels=[ds.rows[i]['exist_label'] for i in selected];values=[preds[i] for i in selected]
    branch[role]={'base_only_auroc':float(roc_auc_score(labels,[p['base_logit'] for p in values])),
     'local_only_auroc':float(roc_auc_score(labels,[p['local_logit'] for p in values])),
     'mean_abs_local_logit':float(np.mean([abs(p['local_logit']) for p in values])),
     'mean_pool_entropy':float(np.mean([p['pool_entropy'] for p in values])),
     'interpretation':'inference decomposition of the same trained model, not retraining control'}
   row['models'][arm]={'seen':seen,'unseen':unseen,'gap':seen['pooled_auroc']-unseen['pooled_auroc'],'seen_val':validation,'best_epoch':saved['epoch'],
     'predictions_sha256':digest(pred_path),'branch_diagnostics':branch,'shuffled_video_retained_label_auroc':shuffled_auc(model,ds,opt)}
   subsets[arm]={'seen':sp,'unseen':up}
   print(a.fold,arm,json.dumps({'seen':seen['pooled_auroc'],'unseen':unseen['pooled_auroc'],'gap':row['models'][arm]['gap']}),flush=True)
   del model,saved;torch.cuda.empty_cache()
  for arm in NEW:
   refs=['B0','S1_saliency_only']+(['Uniform_CF','Local_noCF'] if arm=='Local_CF' else [])
   row['comparisons'][arm]={}
   for ref in refs:
    comparison={}
    for role in ['seen','unseen']:
     comparison[role]={'delta_auroc':row['models'][arm][role]['pooled_auroc']-row['models'][ref][role]['pooled_auroc'],**interval(subsets[ref][role],subsets[arm][role])}
    comparison['gap_reduction']=row['models'][ref]['gap']-row['models'][arm]['gap'];row['comparisons'][arm][ref]=comparison
  write(dest/'RESULTS.json',row)
  lines=[f'# {a.fold}: Moment local evidence, exploratory Unseen','',
   'Seed3407,10epochs, Seen-val selected/thresholded; references from immutable same-budget v2 predictions.','',
   '| Arm | Seen | U | Gap(pp) | ΔU vs B0(pp) | ΔU vs S1(pp) | U CI vs S1(pp) |','|---|---:|---:|---:|---:|---:|---|']
  for arm,m in row['models'].items():
   b=row['models']['B0'];s=row['models']['S1_saliency_only'];ci=row['comparisons'].get(arm,{}).get('S1_saliency_only',{}).get('unseen',{}).get('delta_ci95')
   text=f'[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]' if ci else '—'
   lines.append(f"| {arm} | {m['seen']['pooled_auroc']:.4f} | {m['unseen']['pooled_auroc']:.4f} | {100*m['gap']:.2f} | {100*(m['unseen']['pooled_auroc']-b['unseen']['pooled_auroc']):+.2f} | {100*(m['unseen']['pooled_auroc']-s['unseen']['pooled_auroc']):+.2f} | {text} |")
  lines+=['','CI is1000 paired video-cluster resamples conditional on one trained seed. Previously examined A1/C1 is exploratory. A smaller gap caused by Seen loss is not sufficient. Conditions, branch decomposition and fresh shuffle diagnostics are in RESULTS.json.']
  (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(dest/'status.json',{'status':'completed','results_sha256':digest(dest/'RESULTS.json')})
 except BaseException as exc:write(dest/'status.json',{'status':'failed','error':repr(exc)});raise
if __name__=='__main__':main()
