import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_flash_saliency import metrics

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
    from train_flash_saliency import start_end_collate,prepare_batch_inputs
    rng=np.random.default_rng(3407);n=len(ds);perm=np.roll(rng.permutation(n),1)
    # Permute videos independently of queries/labels while retaining original video masks.
    items=[]
    for i,j in enumerate(perm):
        sample=ds[i];mixed=dict(sample[1]);mixed['video_feat']=ds[int(j)][1]['video_feat'];items.append((sample[0],mixed))
    scores=[]
    for metas,batch in DataLoader(items,batch_size=1,shuffle=False,num_workers=0,collate_fn=start_end_collate):
        inputs,_=prepare_batch_inputs(batch,opt.device);scores.extend(model(**inputs,targets={'label':metas})['pred_exist_logits'].cpu().tolist())
    scores=np.array(scores)
    return {part:float(roc_auc_score([r['exist_label'] for r in ds.rows if r['partition'].startswith(part)],
                scores[[i for i,r in enumerate(ds.rows) if r['partition'].startswith(part)]])) for part in ['S','U']}
