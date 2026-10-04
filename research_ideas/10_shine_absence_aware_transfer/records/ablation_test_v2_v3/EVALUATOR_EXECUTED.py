"""Frozen selected checkpoint exploration on formal A1/C1; no model selection."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from train_formal import ROOT,CachedDataset,build_model,evaluate,read,write,digest
from evaluate_formal_subset import subset_diagnostic,shuffled_auc
from summarize_results import interval
ARMS={'v2':['B0','S1_saliency_only','E1_rotated_exist_only','SE1_full'],
      'v3':['BCE_only','Pair_only','Weak_BCE_pair']}
BASE=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2')
OUT=ROOT/'records/ablation_test_v2_v3'

def gate(split):
    freeze=json.loads((OUT/'EVALUATION_FREEZE.json').read_text())
    for name,h in freeze['file_sha256'].items():
        if digest(name)!=h:raise RuntimeError(f'Evaluation freeze mismatch {name}')
    return freeze['checkpoints'][split]

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);p.add_argument('--device',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    identities=gate(a.fold);destination=OUT/a.fold
    destination.mkdir(exist_ok=False)
    write(destination/'status.json',{'status':'running','scope':'exploratory formal test; no selection'})
    try:
        ck=torch.load(BASE/a.fold/'moment/best.ckpt',map_location='cpu',weights_only=False)
        opt=copy.deepcopy(ck['opt']);opt.device=a.device
        ds=CachedDataset(opt,a.fold,'test');seen_ds=CachedDataset(opt,a.fold,'seen_val')
        row={'split':a.fold,'exploratory':True,'test_used_for_selection':False,'test_sha256':digest(ROOT/f'data/formal/{a.fold}/test.jsonl'),
             'models':{},'comparisons':{},'checkpoints':identities}
        subsets={}
        for arm,identity in identities.items():
            path=identity['path'];saved=torch.load(path,map_location='cpu',weights_only=False)
            model,_=build_model(opt);model.load_state_dict(saved['model'],strict=True);model.to(a.device).eval()
            validation,_=evaluate(model,seen_ds,opt);threshold=validation['threshold_logit']
            predpath=destination/(arm+'_test_predictions.jsonl')
            evaluate(model,ds,opt,threshold=threshold,save=predpath);preds=read(predpath)
            seen,sr,sp=subset_diagnostic(ds.rows,preds,'S',threshold)
            unseen,ur,up=subset_diagnostic(ds.rows,preds,'U',threshold)
            row['models'][arm]={'seen':seen,'unseen':unseen,'gap':seen['pooled_auroc']-unseen['pooled_auroc'],
                    'seen_val':validation,'best_epoch':identity['selected_epoch'],'predictions_sha256':digest(predpath),
                    'shuffled_video_retained_label_auroc':shuffled_auc(model,ds,opt)}
            subsets[arm]={'seen':sp,'unseen':up}
            print(a.fold,arm,json.dumps({'seen':seen['pooled_auroc'],'unseen':unseen['pooled_auroc']}),flush=True)
            del model,saved;torch.cuda.empty_cache()
        for arm in row['models']:
            if arm in ('canonical','B0'):continue
            row['comparisons'][arm]={}
            for reference in ['B0','S1_saliency_only']:
                if reference==arm:continue
                comparison={}
                for role in ['seen','unseen']:
                    comparison[role]={'delta_auroc':row['models'][arm][role]['pooled_auroc']-row['models'][reference][role]['pooled_auroc'],
                            **interval(subsets[reference][role],subsets[arm][role])}
                comparison['gap_reduction']=row['models'][reference]['gap']-row['models'][arm]['gap']
                row['comparisons'][arm][reference]=comparison
        write(destination/'RESULTS.json',row)
        lines=[f'# {a.fold}: v2/v3 exploratory formal test','',
               'All adapted arms: 10 epochs, seed3407, selected only by Seen-val. A1/C1 test already examined; exploratory results.','',
               '| Arm | Epoch | Seen | Unseen | ΔSeen vs B0 (pp) | ΔUnseen vs B0 (pp) | U 95% CI (pp) |',
               '|---|---:|---:|---:|---:|---:|---|']
        b=row['models']['B0']
        for arm,m in row['models'].items():
            c=row['comparisons'].get(arm,{}).get('B0',{}).get('unseen',{}).get('delta_ci95')
            ci=f'[{100*c[0]:+.2f}, {100*c[1]:+.2f}]' if c else '—'
            lines.append(f"| {arm} | {m['best_epoch']} | {m['seen']['pooled_auroc']:.4f} | {m['unseen']['pooled_auroc']:.4f} | {100*(m['seen']['pooled_auroc']-b['seen']['pooled_auroc']):+.2f} | {100*(m['unseen']['pooled_auroc']-b['unseen']['pooled_auroc']):+.2f} | {ci} |")
        lines+=['','CI: 1000 paired video-cluster resamples for fixed trained seed, not retraining uncertainty. Gap reduction accompanied by Seen damage does not establish success. Conditional ranking, localization and shuffled-video diagnostics are in RESULTS.json.']
        (destination/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        write(destination/'status.json',{'status':'completed','results_sha256':digest(destination/'RESULTS.json')})
    except BaseException as exc:
        write(destination/'status.json',{'status':'failed','error':repr(exc)});raise
if __name__=='__main__':main()
