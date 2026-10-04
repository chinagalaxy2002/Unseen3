"""Post-50-epoch exploration of canonical FlashVTG, matched B0 and SHINE saliency."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from train_flash_saliency import ROOT,canonical_path,CachedDataset,build_model,evaluate,read,write,digest
from evaluation_helpers import subset_diagnostic,shuffled_auc
from bootstrap_helpers import interval
BASE=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2')

def gate(split):
    freeze=json.loads((ROOT/'configs/EXPERIMENT_FREEZE.json').read_text())
    for name,h in freeze['file_sha256'].items():
        if digest(name)!=h:raise RuntimeError(f'Input freeze mismatch {name}')
    paths={'canonical':canonical_path(split)};histories=[];ids={}
    for arm in ['B0','S1_saliency_only']:
        run=ROOT/f'runs/formal50/{split}/{arm}/seed3407/flash_saliency_v1_50ep'
        state=json.loads((run/'status.json').read_text());h=json.loads((run/'history.json').read_text());m=json.loads((run/'metrics.json').read_text())
        if state['status']!='completed' or state['epochs_done']!=50 or len(h)!=50:raise RuntimeError('Incomplete '+str(run))
        best=max(x['seen']['pooled_auroc'] for x in h);ep=next(x['epoch'] for x in h if x['seen']['pooled_auroc']==best)
        if ep!=m['best_epoch'] or best!=m['seen_val']['pooled_auroc']:raise RuntimeError('Selection rule mismatch')
        if digest(run/'best.ckpt')!=state['checkpoint_sha256']:raise RuntimeError('Checkpoint mismatch')
        paths[arm]=run/'best.ckpt';histories.append([x['exposure_sha256'] for x in h]);ids[arm]={'path':str(paths[arm]),'sha256':state['checkpoint_sha256'],'selected_epoch':ep}
    if histories[0]!=histories[1]:raise RuntimeError('Forward/exposure control mismatch')
    ids['canonical']={'path':str(paths['canonical']),'sha256':digest(paths['canonical']),'selected_epoch':'canonical'}
    out=ROOT/'records/evaluation';out.mkdir(parents=True,exist_ok=True)
    p=out/f'{split}_EVALUATION_FREEZE.json'
    with p.open('x') as f:json.dump({'created_unix':time.time(),'checkpoints':ids,'formal_U_labels_read_at_gate':False,'test_sha256':digest(ROOT/f'data/formal/{split}/test.jsonl'),'evaluator_sha256':digest(__file__),'training_freeze_sha256':digest(ROOT/'configs/EXPERIMENT_FREEZE.json'),'all_50_exposure_hashes_match':True,'selection':'Seen-val only; earliest trained best; no epoch0 fallback'},f,indent=2)
    return paths,ids

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);p.add_argument('--device',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    paths,ids=gate(a.fold);out=ROOT/'records/evaluation'/a.fold;out.mkdir(exist_ok=False)
    write(out/'status.json',{'status':'running'})
    try:
        source=torch.load(paths['canonical'],map_location='cpu',weights_only=False);opt=copy.deepcopy(source['opt']);opt.device=a.device
        test=CachedDataset(opt,a.fold,'test');val=CachedDataset(opt,a.fold,'seen_val')
        result={'split':a.fold,'backbone':'FlashVTG-GMR','seed':3407,'adaptation_epochs':50,'exploratory':True,'test_used_for_selection':False,'checkpoints':ids,'models':{},'comparisons':{}};predictions={}
        for arm,path in paths.items():
            saved=torch.load(path,map_location='cpu',weights_only=False);model,_=build_model(opt);model.load_state_dict(saved['model'],strict=True);model.to(a.device).eval()
            seenval,_=evaluate(model,val,opt);threshold=seenval['threshold_logit'];p=out/(arm+'_test_predictions.jsonl')
            evaluate(model,test,opt,threshold=threshold,save=p);preds=read(p)
            seen,sr,sp=subset_diagnostic(test.rows,preds,'S',threshold);unseen,ur,up=subset_diagnostic(test.rows,preds,'U',threshold)
            result['models'][arm]={'seen':seen,'unseen':unseen,'gap':seen['pooled_auroc']-unseen['pooled_auroc'],'seen_val':seenval,'selected_epoch':ids[arm]['selected_epoch'],'predictions_sha256':digest(p),'shuffled_video_retained_label_auroc':shuffled_auc(model,test,opt)}
            predictions[arm]={'seen':sp,'unseen':up};print(a.fold,arm,seen['pooled_auroc'],unseen['pooled_auroc'],flush=True)
            del saved,model;torch.cuda.empty_cache()
        method=result['models']['S1_saliency_only']
        for ref in ['canonical','B0']:
            base=result['models'][ref];c={}
            for role in ['seen','unseen']:
                c[role]={'delta_auroc':method[role]['pooled_auroc']-base[role]['pooled_auroc'],**interval(predictions[ref][role],predictions['S1_saliency_only'][role])}
            c['gap_reduction']=base['gap']-method['gap'];c['relative_gap_reduction']=c['gap_reduction']/base['gap'] if base['gap'] else None
            result['comparisons']['S1_minus_'+ref]=c
        write(out/'RESULTS.json',result)
        lines=[f'# FlashVTG-GMR {a.fold}: 50 epochs','', 'canonical initialization; seed3407; Seen-val selection; exploratory formal test.','',
               '| Arm | Seen | Unseen | Gap |','|---|---:|---:|---:|']
        for arm,m in result['models'].items():lines.append(f"| {arm} | {m['seen']['pooled_auroc']:.4f} | {m['unseen']['pooled_auroc']:.4f} | {m['gap']:.4f} |")
        c=result['comparisons']['S1_minus_B0'];lo,hi=c['unseen']['delta_ci95'];lines+=['',f"S1−B0: ΔSeen {100*c['seen']['delta_auroc']:+.2f}pp; ΔUnseen {100*c['unseen']['delta_auroc']:+.2f}pp, paired video CI [{100*lo:+.2f},{100*hi:+.2f}]pp; gap reduction {100*c['gap_reduction']:+.2f}pp.", '', 'Gap reduction from Seen damage alone is not success. Conditions, raw/gated localization and fresh shuffle are in RESULTS.json. CI is conditional on a single trained seed.']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(out/'status.json',{'status':'completed','results_sha256':digest(out/'RESULTS.json')})
    except BaseException as exc:write(out/'status.json',{'status':'failed','error':repr(exc)});raise
if __name__=='__main__':main()
