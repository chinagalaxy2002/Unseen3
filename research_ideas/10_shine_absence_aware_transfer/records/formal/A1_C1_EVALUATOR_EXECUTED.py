"""Scoped formal U evaluation; selected splits must finish both frozen 50-epoch arms."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_formal import ROOT,CachedDataset,build_model,evaluate,metrics,read,write,digest
from summarize_results import interval

SPLITS=['A1','C1']
SCOPE='A1_C1'
BASE=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2')
RELEASE=Path('/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2')


def check_training():
    freeze=json.loads((ROOT/'configs/FORMAL_FREEZE.json').read_text())
    for name,checksum in freeze['code_sha256'].items():
        if digest(ROOT/name)!=checksum:raise RuntimeError('Changed frozen code: '+name)
    checkpoints={}
    for split in SPLITS:
        checkpoints[split]={}
        for arm in ['baseline','shine']:
            run=ROOT/'runs/formal'/split/arm/'seed3407/formal_v1_50ep'
            status=json.loads((run/'status.json').read_text())
            if status['status']!='completed' or status['epochs_done']!=50:raise RuntimeError('Incomplete training: '+str(run))
            checksum=digest(run/'best.ckpt')
            if checksum!=status['checkpoint_sha256']:raise RuntimeError('Checkpoint mismatch')
            checkpoints[split][arm]=checksum
        checkpoints[split]['canonical']=digest(BASE/split/'moment/best.ckpt')
        if checkpoints[split]['canonical']!=freeze['data'][split]['checkpoint_sha256']:raise RuntimeError('Canonical checkpoint changed')
    write(ROOT/'records/formal/EVALUATION_CHECKPOINT_FREEZE.json',{'checkpoints':checkpoints,'freeze_sha256':digest(ROOT/'configs/FORMAL_FREEZE.json'),
          'selection':'Seen-val only','formal_U_read_at_this_gate':False,'evaluated_splits':SPLITS,'evaluator_sha256':digest(Path(__file__)),'scope':'requested partial formal evaluation'})
    return checkpoints


def subset_diagnostic(rows,predictions,partition,threshold):
    selected=[i for i,r in enumerate(rows) if r['partition'].startswith(partition)]
    rs=[rows[i] for i in selected];ps=[predictions[i] for i in selected];scores=np.array([p['logit'] for p in ps])
    result=metrics(rs,scores);hit=[];accept=[]
    for row,pred in zip(rs,ps):
        if not row['exist_label']:continue
        a,b=pred['span'];overlaps=[]
        for start,end in row['relevant_windows']:
            inter=max(0,min(b,end)-max(a,start));union=max(1e-9,b-a+end-start-inter)
            overlaps.append(inter/union)
        hit.append(max(overlaps,default=0)>=.5);accept.append(pred['logit']>=threshold)
    result.update(raw_r1_iou_05=float(np.mean(hit)),gated_r1_iou_05=float(np.mean(np.array(hit)&np.array(accept))),
        positive_count=sum(r['exist_label'] for r in rs),negative_count=sum(not r['exist_label'] for r in rs),
        query_count=len({r['query'] for r in rs}),threshold_logit=threshold)
    return result,rs,ps


@torch.no_grad()
def shuffled_auc(model,ds,opt):
    # Fresh pre-fusion re-forward. Retained labels diagnose dependence only.
    from torch.utils.data import DataLoader
    from train_formal import start_end_collate,prepare_batch_inputs
    rng=np.random.default_rng(3407);n=len(ds);perm=np.roll(rng.permutation(n),1)
    # Permute videos independently of queries/labels while retaining original video masks.
    items=[]
    for i,j in enumerate(perm):
        sample=ds[i];mixed={'meta':sample['meta'],'model_inputs':dict(sample['model_inputs'])}
        mixed['model_inputs']['video_feat']=ds[int(j)]['model_inputs']['video_feat'];items.append(mixed)
    scores=[]
    for _,batch in DataLoader(items,batch_size=32,shuffle=False,num_workers=0,collate_fn=start_end_collate):
        inputs,_=prepare_batch_inputs(batch,opt.device);scores.extend(model(**inputs)['pred_exist_logits'].cpu().tolist())
    scores=np.array(scores)
    return {part:float(roc_auc_score([r['exist_label'] for r in ds.rows if r['partition'].startswith(part)],
                scores[[i for i,r in enumerate(ds.rows) if r['partition'].startswith(part)]])) for part in ['S','U']}


def run(device):
    identities=check_training()
    out=ROOT/'records/formal';results=[]
    for split in SPLITS:
        test_src=RELEASE/split/'test.jsonl';local=ROOT/'data/formal'/split/'test.jsonl'
        shutil.copyfile(test_src,local)
        source=torch.load(BASE/split/'moment/best.ckpt',map_location='cpu',weights_only=False)
        opt=copy.deepcopy(source['opt']);opt.device=device
        ds=CachedDataset(opt,split,'test');seen_ds=CachedDataset(opt,split,'seen_val')
        paths={'canonical':BASE/split/'moment/best.ckpt'}
        paths.update({arm:ROOT/'runs/formal'/split/arm/'seed3407/formal_v1_50ep/best.ckpt' for arm in ['baseline','shine']})
        models={};row={'split':split,'test_source':str(test_src),'test_sha256':digest(test_src),'models':{},'comparisons':{},
            'historical_canonical':json.loads((BASE/split/'moment/diagnostics.json').read_text()),'checkpoints':identities[split]}
        for arm,path in paths.items():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model,_=build_model(opt);model.load_state_dict(saved['model'],strict=True);model.to(device).eval()
            validation,_=evaluate(model,seen_ds,opt);threshold=validation['threshold_logit']
            prediction_path=out/(split+'_'+arm+'_test_predictions.jsonl')
            _,_=evaluate(model,ds,opt,threshold=threshold,save=prediction_path)
            preds=read(prediction_path)
            seen,sr,sp=subset_diagnostic(ds.rows,preds,'S',threshold)
            unseen,ur,up=subset_diagnostic(ds.rows,preds,'U',threshold)
            shuffle=shuffled_auc(model,ds,opt)
            row['models'][arm]={'seen':seen,'unseen':unseen,'gap':seen['pooled_auroc']-unseen['pooled_auroc'],
                'seen_val':validation,'best_epoch':saved.get('epoch'),'predictions_sha256':digest(prediction_path),
                'shuffled_video_retained_label_auroc':shuffle}
            models[arm]={'S':(sr,sp),'U':(ur,up)}
            del model,saved
            torch.cuda.empty_cache()
        for comparator in ['canonical','baseline']:
            comparison={}
            for key,prefix in [('seen','S'),('unseen','U')]:
                base_rows,base_preds=models[comparator][prefix];method_rows,method_preds=models['shine'][prefix]
                comparison[key]={'delta_auroc':row['models']['shine'][key]['pooled_auroc']-row['models'][comparator][key]['pooled_auroc'],
                    **interval(base_preds,method_preds)}
            comparison['gap_reduction']=row['models'][comparator]['gap']-row['models']['shine']['gap']
            row['comparisons']['shine_minus_'+comparator]=comparison
        write(out/(split+'_FORMAL_RESULTS.json'),row);results.append(row)
        print(split,{arm:{'seen':v['seen']['pooled_auroc'],'unseen':v['unseen']['pooled_auroc'],'gap':v['gap']} for arm,v in row['models'].items()},flush=True)
    macro={arm:{metric:float(np.mean([r['models'][arm][metric]['pooled_auroc'] for r in results])) for metric in ['seen','unseen']} for arm in ['canonical','baseline','shine']}
    for arm in macro:macro[arm]['gap']=macro[arm]['seen']-macro[arm]['unseen']
    report={'status':'completed_requested_subset','evaluated_splits':SPLITS,'five_split_complete':False,'backbone':'Moment-DETR-GMR','seed':3407,'adaptation_epochs':50,'splits':results,'macro':macro,
        'common_protocol_historical_reference':{'seen':.7518,'unseen':.5287,'gap':.2231},'official_U_evaluated':True,
        'test_used_for_selection':False,'interpretation':'exploratory formal replication; full precision raw logits',
        'missing_mechanism_controls':['matched five-split Cq/Cv retraining','independently verified closed quartets'],
        'other_backbones_evaluated':False}
    write(out/(SCOPE+'_FORMAL_RESULTS.json'),report)
    lines=['# Idea 10：正式 Unseen 退化缓解结果（50 epochs）','',
        'Moment-DETR-GMR，seed 3407，本次仅评测 A1、C1；macro 是这两个 splits 的等权平均，不代表五 split。新训练两 arms 各 50 epochs，从同 canonical checkpoint 初始化；只按 Seen-val AUROC 选模，正式 U 仅在训练完成且 checkpoint 冻结后评测。', '',
        '| Split | Arm | Seen AUROC | Unseen AUROC | Gap |','|---|---|---:|---:|---:|']
    for r in results:
        for arm,m in r['models'].items():lines.append(f"| {r['split']} | {arm} | {m['seen']['pooled_auroc']:.4f} | {m['unseen']['pooled_auroc']:.4f} | {m['gap']:.4f} |")
    for arm,m in macro.items():lines.append(f"| **Macro** | **{arm}** | **{m['seen']:.4f}** | **{m['unseen']:.4f}** | **{m['gap']:.4f}** |")
    lines+=['','| Comparator | A1/C1 Macro ΔUnseen (pp) | A1/C1 Macro ΔSeen (pp) | Gap reduction (pp) |','|---|---:|---:|---:|']
    for arm in ['canonical','baseline']:
        lines.append(f"| SHINE−{arm} | {(macro['shine']['unseen']-macro[arm]['unseen'])*100:+.2f} | {(macro['shine']['seen']-macro[arm]['seen'])*100:+.2f} | {(macro[arm]['gap']-macro['shine']['gap'])*100:+.2f} |")
    lines+=['','COMMON_PROTOCOL 的 Moment 历史参考为 Seen .7518 / Unseen .5287 / Gap .2231；canonical 行为同 checkpoint 的完整精度重算。历史分数精度/选模口径不同处不能当作新方法收益。', '',
        'Gap 缩小若伴随 Seen 降低，不能单独算退化缓解；主判断是 ΔUnseen 与 Seen 保持。不同 split 共享视频，未给出假设 split 独立的 macro CI。逐 split 共享视频配对区间、条件排序、raw/gated top1 定位和 fresh pre-fusion shuffle 见 [A1_C1_FORMAL_RESULTS.json](A1_C1_FORMAL_RESULTS.json)。', '',
        '这里只迁移 Moment，不能声称 Flash/QD 的 .5479/.5144 退化已缓解；Cq/Cv 与闭合四格对照仍缺，不能将 pooled 收益直接归因于真正事件证据。没有运行多 seed。']
    (out/(SCOPE+'_FORMAL_RESULTS.md')).write_text('\n'.join(lines)+'\n')
    print('MACRO '+json.dumps(macro),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.set_num_threads(4);run(args.device)
