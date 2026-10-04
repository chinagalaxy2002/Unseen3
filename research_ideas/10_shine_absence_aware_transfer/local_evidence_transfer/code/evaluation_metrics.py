import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_local_evidence import metrics

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
    from train_local_evidence import start_end_collate,prepare_batch_inputs
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

def interval(base,method):
    if [(r['qid'],r['vid'],r['exist_label']) for r in base] != [(r['qid'],r['vid'],r['exist_label']) for r in method]:
        raise RuntimeError('Prediction identity mismatch')
    y=np.array([r['exist_label'] for r in base])
    b=np.array([r['logit'] for r in base]);m=np.array([r['logit'] for r in method])
    _,inverse=np.unique([r['vid'] for r in base],return_inverse=True)
    n=inverse.max()+1;rng=np.random.default_rng(3407);deltas=[]
    for _ in range(1000):
        weights=np.bincount(rng.integers(n,size=n),minlength=n)[inverse]
        if any(weights[y==label].sum()==0 for label in [0,1]):continue
        deltas.append(roc_auc_score(y,m,sample_weight=weights)-roc_auc_score(y,b,sample_weight=weights))
    return {'replicates':len(deltas),'unit':'paired video cluster','delta_ci95':np.quantile(deltas,[.025,.975]).tolist()}
