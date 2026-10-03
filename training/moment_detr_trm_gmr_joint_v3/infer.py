"""Seen-only calibration followed by full-precision four-partition evaluation."""
from __future__ import annotations
import argparse,json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'training/moment_detr_gmr'))
from models.moment_detr_gmr.utils.basic_utils import load_jsonl,save_jsonl
from training.moment_detr_trm_gmr_joint_v3.evaluate import compute_mr_results_joint,setup_model_joint
from training.moment_detr_trm_gmr_joint_v3.dataset import StartEndDatasetJoint,start_end_collate_joint
from training.moment_detr_trm_gmr_joint_v3.train import build_dataset_config_joint
from training.moment_detr_trm_gmr_joint_v3.semantic_groups import group_auroc
from scripts.analyze_semantic_existence import choose_threshold,iou


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def localization(rows,byid,key):
    values=[]
    for r in rows:
        windows=byid[str(r['qid'])].get(key,[])
        values.append(max((iou(windows[0][:2],gt) for gt in r['relevant_windows']),default=0.0) if windows else 0.0)
    arr=np.asarray(values)
    return {'count':len(values),'R1@0.5':float(np.mean(arr>=.5)*100) if len(arr) else None,'mIoU':float(np.mean(arr)*100) if len(arr) else None}


def compute_auc(rows,byid):
    y=np.asarray([int(r['partition'].endswith('+')) for r in rows]);scores=np.asarray([byid[str(r['qid'])]['pred_exist_logit'] for r in rows])
    probs=np.asarray([byid[str(r['qid'])]['pred_exist_score'] for r in rows])
    if not np.isfinite(scores).all() or not np.isfinite(probs).all(): raise FloatingPointError('NaN/Inf prediction')
    logit_auc=float(roc_auc_score(y,scores)); prob_auc=float(roc_auc_score(y,probs))
    if abs(logit_auc-prob_auc)>1e-12: raise ValueError(f'Logit/probability AUROC mismatch: {logit_auc} vs {prob_auc}; inspect saturation/ties')
    return logit_auc


def run(args):
    out=Path(args.results_dir);release=Path(args.release_dir)
    meta=json.loads((out/'training_meta.json').read_text())
    if meta['best_epoch']<1:raise ValueError('No localization-constrained checkpoint: U inference forbidden')
    if not (out/'best_selection.json').exists():raise ValueError('Missing eligible Seen selection record')
    selection=json.loads((out/'best_selection.json').read_text())
    if not selection['localization_constraint_satisfied']:raise ValueError('Checkpoint fails localization constraint')
    val=[r for r in load_jsonl(str(release/'val.jsonl')) if r['partition'] in ('S+','S-')]
    predictions=load_jsonl(str(out/'best_charades_sta_semantic_novelty_val_preds.jsonl'))
    byid={str(p['qid']):p for p in predictions}
    if {str(r['qid']) for r in val} != set(byid):raise ValueError('Missing/extra Seen-val predictions')
    threshold=choose_threshold(val,byid)
    save_jsonl(predictions,str(out/'val_predictions.jsonl'))
    frozen={'threshold':threshold,'threshold_source':'Seen validation S+/S- maximum balanced accuracy',
            'best_epoch':meta['best_epoch'],'best_seen_val_mAP':meta['best_seen_val_mAP'],
            'selection_metric':meta['selection_metric'],'best_seen_worst_semantic_auroc':meta['best_seen_worst_semantic_auroc'],
            'localization_floor_mAP':meta['localization_floor_mAP'],'val_seen_n':len(val)}
    write_json(out/'threshold_frozen.json',frozen)
    # Only after the Seen checkpoint and threshold are frozen may U be read.
    test=load_jsonl(str(release/'test.jsonl'))
    pairs=load_jsonl(str(release/'matched_u_pairs.jsonl'))
    checkpoint=torch.load(args.model_path,map_location='cpu',weights_only=False)
    opt=checkpoint['opt'];opt.device=args.device;opt.results_dir=str(out);opt.exist_gate_thd=threshold;opt.eval_bsz=16
    model=setup_model_joint(opt);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    dataset=StartEndDatasetJoint(**build_dataset_config_joint(opt,str(release/'test.jsonl'),load_labels=False,keep_empty_gt=True))
    if len(dataset)!=len(test):raise ValueError('Missing test feature; fallback forbidden')
    loader=DataLoader(dataset,batch_size=16,num_workers=opt.num_workers,collate_fn=start_end_collate_joint,shuffle=False)
    pred,_=compute_mr_results_joint(-1,model,loader,opt)
    save_jsonl(pred,str(out/'test_predictions.jsonl'));byid={str(p['qid']):p for p in pred}
    if set(byid)!={str(r['qid']) for r in test}:raise ValueError('Incomplete full-test predictions')
    seen=[r for r in test if r['partition'] in ('S+','S-')];unseen=[r for r in test if r['partition'] in ('U+','U-')]
    seen_auc=compute_auc(seen,byid);unseen_auc=compute_auc(unseen,byid)
    quadrants={}; loc={}
    for part in ['S+','S-','U+','U-']:
        rows=[r for r in test if r['partition']==part];sc=np.asarray([byid[str(r['qid'])]['pred_exist_score'] for r in rows])
        d={'n':len(rows),'mean_exist_score':float(sc.mean())}
        if part.endswith('+'):
            d['false_refusal']=float(np.mean(sc<threshold))
            loc[part]={'raw':localization(rows,byid,'pred_relevant_windows_pre_exist'),'official_gated':localization(rows,byid,'pred_relevant_windows')}
        else:d['rejection_rate']=float(np.mean(sc<threshold))
        quadrants[part]=d
    pair_values=[]
    for p in pairs:
        positive=byid[str(p['positive_qid'])]['pred_exist_logit'];negative=byid[str(p['negative_qid'])]['pred_exist_logit']
        pair_values.append(float(positive>negative)+.5*float(positive==negative))
    pairacc=float(np.mean(pair_values)) if pair_values else None
    v1=json.loads((ROOT/'results/moment_detr_trm_gmr_joint'/release.name/'joint_summary.json').read_text())
    v2_path=ROOT/'results/moment_detr_trm_gmr_joint_v2'/release.name/'joint_v2_summary.json'
    v2=json.loads(v2_path.read_text()) if v2_path.exists() else {}
    summary={'split':release.name,'model':'Moment-DETR-TRM-GMR-Joint-v3','status':'completed',
      'exploratory_status':'Exploratory follow-up motivated by previously inspected Joint-v1/v2 failure modes',
      'seen_auroc':seen_auc,'unseen_auroc':unseen_auc,'seen_unseen_gap':seen_auc-unseen_auc,
      'joint_v1_unseen_auroc':v1['unseen_auroc'],'delta_v3_vs_v1_unseen_auroc':unseen_auc-v1['unseen_auroc'],
      'joint_v2_unseen_auroc':v2.get('unseen_auroc'),
      'delta_v3_vs_v2_unseen_auroc':unseen_auc-v2['unseen_auroc'] if v2.get('unseen_auroc') is not None else None,
      'S+_raw_R1@0.5':loc['S+']['raw']['R1@0.5'],'U+_raw_R1@0.5':loc['U+']['raw']['R1@0.5'],
      'S+_raw_mIoU':loc['S+']['raw']['mIoU'],'U+_raw_mIoU':loc['U+']['raw']['mIoU'],
      'U+_FRR':quadrants['U+']['false_refusal'],'U-_RR':quadrants['U-']['rejection_rate'],
      'matched_pair_acc':pairacc,'best_epoch':meta['best_epoch'],'threshold':threshold,
      'best_seen_worst_semantic_auroc':meta['best_seen_worst_semantic_auroc'],
      'best_seen_val_mAP':meta['best_seen_val_mAP'],'localization_floor_mAP':meta['localization_floor_mAP'],
      'epochs_trained':meta['epochs_trained'],'auc_valid_batch_fraction':meta['auc_valid_batch_fraction'],
      'nan_inf_detected':False,'missing_feature_or_fallback':False,'pca_used':False,
      'counts':{k:v['n'] for k,v in quadrants.items()}}
    diagnostics={'quadrants':quadrants,'raw_and_official_localization':loc,'matched_pair_n':len(pair_values),
      'Seen_test_groups':group_auroc(seen,pred)}
    # U group metrics are descriptive only; never used by training/selection.
    from training.moment_detr_trm_gmr_joint_v3.semantic_groups import semantic_keys
    u_for_groups=[dict(r,partition='S+' if r['partition']=='U+' else 'S-') for r in unseen]
    diagnostics['Unseen_test_groups']=group_auroc(u_for_groups,pred)
    up=[r for r in test if r['partition']=='U+']
    hardhits=[byid[str(r['qid'])]['pred_exist_score']>=threshold and max((iou(byid[str(r['qid'])]['pred_relevant_windows_pre_exist'][0][:2],gt) for gt in r['relevant_windows']),default=0)>=.5 for r in up]
    diagnostics['diagnostic_hard_gate_U+_R1@0.5']=float(np.mean(hardhits)*100)
    write_json(out/'diagnostics.json',diagnostics)
    subprocess.run([sys.executable,str(ROOT/'eval/eval_main.py'),'--submission_path',str(out/'test_predictions.jsonl'),'--gt_path',str(release/'test.jsonl'),'--save_path',str(out/'official_test_metrics.json')],check=True,cwd=ROOT)
    write_json(out/'joint_v3_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model_path',required=True);p.add_argument('--release_dir',required=True);p.add_argument('--results_dir',required=True);p.add_argument('--device',default='cuda');run(p.parse_args())
