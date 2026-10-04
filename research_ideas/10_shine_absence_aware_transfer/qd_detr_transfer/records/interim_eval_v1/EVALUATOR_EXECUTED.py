"""User-authorized interim evaluation of immutable QD snapshots; training continues."""
import argparse
import copy
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"code"))
import torch
from train_qd_saliency import ROOT,CachedDataset,build_model,evaluate,read,write,digest
from evaluation_helpers import subset_diagnostic,shuffled_auc
from bootstrap_helpers import interval
BASE=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2')

INTERIM=Path(__file__).resolve().parent

def gate(split):
    freeze=json.loads((INTERIM/'FREEZE.json').read_text())
    for name,h in freeze['file_sha256'].items():
        if digest(name)!=h:raise RuntimeError('Interim snapshot/input mismatch '+name)
    paths={arm:Path(row['path']) for arm,row in freeze['checkpoints'][split].items()}
    return paths,freeze['checkpoints'][split]


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);p.add_argument('--device',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    paths,ids=gate(a.fold);out=INTERIM/a.fold;out.mkdir(exist_ok=False)
    write(out/'status.json',{'status':'running'})
    try:
        source=torch.load(paths['canonical'],map_location='cpu',weights_only=False);opt=copy.deepcopy(source['opt']);opt.device=a.device
        test=CachedDataset(opt,a.fold,'test');val=CachedDataset(opt,a.fold,'seen_val')
        result={'split':a.fold,'backbone':'QD-DETR-GMR','seed':3407,'interim':True,'full_50_epoch_complete':False,'adaptation_epochs':None,'training_epochs_observed':{arm:row['epochs_observed'] for arm,row in ids.items()},'exploratory':True,'test_used_for_selection':False,'checkpoints':ids,'models':{},'comparisons':{}};predictions={}
        for arm,path in paths.items():
            saved=torch.load(path,map_location='cpu',weights_only=False);model,_=build_model(opt);model.load_state_dict(saved['model'],strict=True);model.to(a.device).eval()
            seenval,_=evaluate(model,val,opt);threshold=seenval['threshold_logit'];p=out/(arm+'_test_predictions.jsonl')
            evaluate(model,test,opt,threshold=threshold,save=p);preds=read(p)
            seen,sr,sp=subset_diagnostic(test.rows,preds,'S',threshold);unseen,ur,up=subset_diagnostic(test.rows,preds,'U',threshold)
            result['models'][arm]={'seen':seen,'unseen':unseen,'gap':seen['pooled_auroc']-unseen['pooled_auroc'],'seen_val':seenval,'selected_epoch':ids[arm]['selected_epoch'],'epochs_observed':ids[arm]['epochs_observed'],'predictions_sha256':digest(p),'shuffled_video_retained_label_auroc':shuffled_auc(model,test,opt)}
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
        lines=[f'# QD-DETR-GMR {a.fold}: interim snapshots','', 'INTERIM: training is still running, budgets differ; seed3407; selected only by Seen-val. This is not the final50-epoch comparison.','',
               '| Arm | Seen | Unseen | Gap |','|---|---:|---:|---:|']
        for arm,m in result['models'].items():lines.append(f"| {arm} | {m['seen']['pooled_auroc']:.4f} | {m['unseen']['pooled_auroc']:.4f} | {m['gap']:.4f} |")
        c=result['comparisons']['S1_minus_B0'];lo,hi=c['unseen']['delta_ci95'];lines+=['',f"S1−B0: ΔSeen {100*c['seen']['delta_auroc']:+.2f}pp; ΔUnseen {100*c['unseen']['delta_auroc']:+.2f}pp, paired video CI [{100*lo:+.2f},{100*hi:+.2f}]pp; gap reduction {100*c['gap_reduction']:+.2f}pp.", '', 'Gap reduction from Seen damage alone is not success. Conditions, raw/gated localization and fresh shuffle are in RESULTS.json. CI is conditional on a single trained seed.']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(out/'status.json',{'status':'completed','results_sha256':digest(out/'RESULTS.json')})
    except BaseException as exc:write(out/'status.json',{'status':'failed','error':repr(exc)});raise
if __name__=='__main__':main()
