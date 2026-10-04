"""A1/C1 matched-forward baseline and saliency-only, 50 epochs.

Derived from frozen train_ablation_v2.py; previous frozen sources unchanged.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

CODE=Path(__file__).resolve().parent
ROOT=CODE.parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(CODE))
from models.moment_detr_gmr.moment_detr import build_model
from models.moment_detr_gmr.utils.basic_utils import l2_normalize_np_array
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
from training.moment_detr_gmr.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from shine_losses import coarse, fine


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CachedDataset(Dataset):
    def __init__(self,opt,fold,role):
        cfg=dict(dset_name=opt.dset_name, domain=None, data_path=str(ROOT/'data/formal'/fold/(role+'.jsonl')),
            v_feat_dirs=opt.v_feat_dirs,q_feat_dir=opt.t_feat_dir, v_feat_types=opt.v_feat_types,
            max_q_l=opt.max_q_l,max_v_l=opt.max_v_l,ctx_mode=opt.ctx_mode,clip_len=opt.clip_length,
            max_windows=opt.max_windows,span_loss_type=opt.span_loss_type,mr_only=True,keep_empty_gt=True)
        ds=StartEndDataset(**cfg)
        if len(ds)!=len(read(cfg['data_path'])):
            raise RuntimeError('Missing features')
        self.rows=ds.data
        # Cache raw normalized video per identity; preserve row-specific queries/labels.
        videos={}
        base=ds._get_video_feat_by_vid
        def get_video(vid):
            if vid not in videos:
                videos[vid]=base(vid)
            return videos[vid]
        ds._get_video_feat_by_vid=get_video
        self.samples=[ds[i] for i in range(len(ds))]
        self.bank=json.loads((ROOT/'data/formal'/fold/'query_bank.json').read_text()) if role=='train' else {}
        self.hn={}
        for qid,entry in self.bank.items():
            arrays=[]
            for edit in entry['edits']:
                with np.load(ROOT/'artifacts/edit_clip'/(edit['feature_id']+'.npz')) as z:
                    a=z['last_hidden_state'][:opt.max_q_l].astype(np.float32)
                arrays.append(torch.from_numpy(l2_normalize_np_array(a)))
            self.hn[qid]=arrays
    def __len__(self): return len(self.samples)
    def __getitem__(self,index): return self.samples[index]


def gt_mask(metas,valid,clip_length):
    gt=torch.zeros_like(valid)
    for i,meta in enumerate(metas):
        length=int(valid[i].sum())
        for start,end in meta.get('relevant_windows',[]):
            lo=max(0,min(int(start/clip_length),length-1))
            hi=max(0,min(int(end/clip_length),length)-1)
            if hi>=lo:
                gt[i,lo:hi+1]=1
    return gt


def pad_hn(ds,metas,indices,device):
    texts=[ds.hn[str(metas[i]['qid'])][level] for level in range(3) for i in indices]
    n=len(texts);length=max(t.shape[0] for t in texts)
    out=torch.zeros(n,length,texts[0].shape[-1],device=device)
    mask=torch.zeros(n,length,device=device)
    for i,t in enumerate(texts):
        out[i,:len(t)]=t.to(device);mask[i,:len(t)]=1
    return out,mask


def metrics(rows,scores):
    labels=np.array([r['exist_label'] for r in rows])
    byid={str(r['qid']):i for i,r in enumerate(rows)}
    pair=[]
    for i,r in enumerate(rows):
        j=byid.get(str(r.get('source_qid')))
        if not r['exist_label'] and j is not None and rows[j]['exist_label'] and rows[j]['vid']==r['vid']:
            d=scores[j]-scores[i];pair.append(float(d>0)+.5*float(d==0))
    groups=defaultdict(lambda:[[],[]])
    for r,s in zip(rows,scores):
        groups[r['query']][r['exist_label']].append(float(s))
    conditional=[]
    for neg,pos in groups.values():
        if pos and neg:
            diff=np.array(pos)[:,None]-np.array(neg)[None,:]
            conditional.append(float(((diff>0)+.5*(diff==0)).mean()))
    return {'pooled_auroc':float(roc_auc_score(labels,scores)),'rows':len(rows),'videos':len({r['vid'] for r in rows}),
        'source_pair_acc':float(np.mean(pair)) if pair else None,'source_pairs':len(pair),
        'same_query_pair_acc':float(np.mean(conditional)) if conditional else None,'same_query_support':len(conditional)}


@torch.no_grad()
def evaluate(model,ds,opt,threshold=None,save=None):
    model.eval()
    scores=[];ious=[];positive_scores=[];predictions=[]
    for metas,batch in DataLoader(ds,batch_size=32,shuffle=False,collate_fn=start_end_collate,num_workers=0):
        inputs,_=prepare_batch_inputs(batch,opt.device)
        out=model(**inputs)
        logits=out['pred_exist_logits'].cpu().numpy()
        probs=out['pred_logits'].softmax(-1)[...,0]
        best=probs.argmax(-1)
        spans=span_cxw_to_xx(out['pred_spans']).cpu().numpy()
        for i,meta in enumerate(metas):
            score=float(logits[i]);scores.append(score)
            span=spans[i,int(best[i])]*meta['duration']
            span=np.clip(span,0,meta['duration'])
            if meta['exist_label']:
                overlaps=[]
                for start,end in meta['relevant_windows']:
                    inter=max(0,min(end,span[1])-max(start,span[0]))
                    union=max(1e-9,end-start+span[1]-span[0]-inter)
                    overlaps.append(inter/union)
                iou=max(overlaps,default=0);ious.append(iou);positive_scores.append(score)
            predictions.append({'qid':meta['qid'],'vid':meta['vid'],'query':meta['query'],'exist_label':meta['exist_label'],
                'source_qid':meta.get('source_qid'),'logit':score,'span':span.tolist()})
    scores=np.array(scores)
    result=metrics(ds.rows,scores)
    result['raw_r1_iou_05']=float(np.mean(np.array(ious)>=.5))
    result['raw_r1_iou_07']=float(np.mean(np.array(ious)>=.7))
    if threshold is None:
        fpr,tpr,thresholds=roc_curve([r['exist_label'] for r in ds.rows],scores)
        finite=np.isfinite(thresholds)
        threshold=float(thresholds[finite][np.argmax((tpr-fpr)[finite])])
    result['threshold_logit']=threshold
    result['gated_r1_iou_05']=float(np.mean((np.array(ious)>=.5)&(np.array(positive_scores)>=threshold)))
    if save:
        with Path(save).open('w') as f:
            for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return result,scores


def run_arm(args,opt,checkpoint,datasets,arm):
    run=ROOT/'runs/saliency_v5'/args.fold/arm/'seed3407'/args.run_id
    run.mkdir(parents=True,exist_ok=True)
    if (run/'status.json').exists():
        raise RuntimeError(f'Existing run: {run}')
    random.seed(3407);np.random.seed(3407);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
    model,criterion=build_model(opt)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.to(opt.device);criterion.to(opt.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=opt.wd)
    opt.results_dir=str(run)
    for key,name in [('ckpt_filepath','best.ckpt'),('train_log_filepath','train.log'),('eval_log_filepath','val.log')]:
        setattr(opt,key,str(run/name))
    write(run/'resolved_config.json',{'saved_options':dict(opt),'ablation':vars(args),'arm':arm,
        'initialization':'same canonical formal checkpoint','selection':'formal Seen-val pooled AUROC; earliest tie',
        'coarse_margin':[1.,2.],'fine_margin':[.25]*4,'fine_type':'red',
        'weights':ARM_WEIGHTS[arm], 'forward_control':'all arms execute identical auxiliary forwards; inactive losses omitted from backward','aux_chain_limit':4,
        'user_attested_batch_absence':True,'user_attested_hierarchy_valid':True,
        'gt_grid':'current GMR nominal clip_length; no independently verified extraction mapping',
        'formal_U_accessed':False})
    start=time.time()
    write(run/'status.json',{'status':'running','started':start})
    try:
        seen0,_=evaluate(model,datasets['seen_val'],opt,save=run/'epoch0_seen.jsonl')
        write(run/'epoch0_metrics.json',{'seen':seen0})
        # Best is a trained epoch; do not silently substitute untouched baseline.
        best=-float('inf');history=[]
        train=datasets['train']
        generator=torch.Generator().manual_seed(3407)
        loader=DataLoader(train,batch_size=opt.bsz,shuffle=True,collate_fn=start_end_collate,num_workers=0,generator=generator)
        for epoch in range(args.epochs):
            model.train();criterion.train();totals=defaultdict(float);steps=0;chains=0;epoch_start=time.time(); exposure=hashlib.sha256()
            for metas,batch in loader:
                exposure.update(json.dumps([str(m['qid']) for m in metas]).encode())
                inputs,targets=prepare_batch_inputs(batch,opt.device)
                out=model(**inputs)
                original=criterion(out,targets)
                loss=sum(v*criterion.weight_dict[k] for k,v in original.items() if k in criterion.weight_dict)
                parts={'gmr':float(loss.detach()), **{'gmr_'+k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in original.items()}}
                if len(metas)>1:
                    rotated=dict(inputs)
                    rotated['src_txt']=inputs['src_txt'].roll(-1,0)
                    rotated['src_txt_mask']=inputs['src_txt_mask'].roll(-1,0)
                    neg=model(**rotated)
                    gt=gt_mask(metas,inputs['src_vid_mask'],opt.clip_length)
                    positive=targets['exist_label'].bool()
                    lc=coarse(out['saliency_scores'],neg['saliency_scores'],gt,inputs['src_vid_mask'],positive)
                    ln=F.binary_cross_entropy_with_logits(neg['pred_exist_logits'],torch.zeros_like(neg['pred_exist_logits']))
                    lp=F.relu(.2-out['pred_exist_logits'][positive]+neg['pred_exist_logits'][positive]).mean() if positive.any() else loss*0
                    ids=[i for i,m in enumerate(metas) if positive[i] and str(m['qid']) in train.hn]
                    if len(ids)>4:
                        ids=random.sample(ids,4)
                    exposure.update(json.dumps(ids).encode())
                    lf=loss*0
                    if ids:
                        text,mask=pad_hn(train,metas,ids,opt.device)
                        extra={'src_txt':text,'src_txt_mask':mask,
                            'src_vid':inputs['src_vid'][ids].repeat(3,1,1),
                            'src_vid_mask':inputs['src_vid_mask'][ids].repeat(3,1)}
                        hard=model(**extra)
                        hn=hard['saliency_scores'].reshape(3,len(ids),-1)
                        lf=fine(out['saliency_scores'][ids],hn,neg['saliency_scores'][ids],gt[ids],inputs['src_vid_mask'][ids])
                        chains+=len(ids)
                    components={'gmr':loss,'coarse':lc,'fine':lf,'rotated_bce':ln,'exist_pair':.2*lp}
                    if steps==0 and epoch+1 in (1,3,10,25,50):
                        write(run/f'gradient_epoch{epoch+1}.json', gradient_probe(model,components,ARM_WEIGHTS[arm]))
                    loss=loss+sum(value*components[key] for key,value in ARM_WEIGHTS[arm].items() if value)
                    parts.update(coarse=float(lc.detach()),fine=float(lf.detach()),rotated_bce=float(ln.detach()),exist_pair=float(lp.detach()))
                parts['total']=float(loss.detach())
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if opt.grad_clip>0:torch.nn.utils.clip_grad_norm_(model.parameters(),opt.grad_clip)
                optimizer.step();steps+=1
                for key,value in parts.items():totals[key]+=value
                if steps%100==0:
                    print(json.dumps({'fold':args.fold,'arm':arm,'epoch':epoch+1,'step':steps,'loss':float(loss.detach())}),flush=True)
            seen,scores=evaluate(model,datasets['seen_val'],opt,save=run/f'epoch{epoch+1}_seen.jsonl')
            labels=np.array([r['exist_label'] for r in datasets['seen_val'].rows])
            seen['score_distribution']={str(label):{'mean':float(scores[labels==label].mean()),'std':float(scores[labels==label].std()),'quantiles':np.quantile(scores[labels==label],[.1,.5,.9]).tolist()} for label in (0,1)}
            selected=seen['pooled_auroc']>best
            row={'epoch':epoch+1,'seen':seen,'losses':{k:v/steps for k,v in totals.items()},'chains':chains,
                'seconds':time.time()-epoch_start,'selected':selected,'exposure_sha256':exposure.hexdigest()}
            history.append(row);write(run/'history.json',history)
            if epoch+1 in (1,3,10,25,50):
                torch.save({'model':model.state_dict(),'opt':opt,'epoch':epoch+1,'arm':arm},run/f'epoch{epoch+1}.ckpt')
            if selected:
                best=seen['pooled_auroc']
                torch.save({'model':model.state_dict(),'opt':opt,'epoch':epoch+1,'arm':arm},run/'best.ckpt')
            print(json.dumps({'fold':args.fold,'arm':arm,**row}),flush=True)
            write(run/'status.json',{'status':'running','epochs_done':epoch+1,'seconds':time.time()-start})
        saved=torch.load(run/'best.ckpt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'])
        seen,seen_scores=evaluate(model,datasets['seen_val'],opt,save=run/'predictions_seen.jsonl')
        result={'fold':args.fold,'arm':arm,'seed':3407,'best_epoch':saved['epoch'],'seen_val':seen,
            'epoch0':{'seen':seen0},'elapsed_seconds':time.time()-start,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'formal_U_accessed':False}
        write(run/'metrics.json',result)
        write(run/'status.json',{'status':'completed','epochs_done':args.epochs,'seconds':time.time()-start,'checkpoint_sha256':digest(run/'best.ckpt')})
        print('COMPLETED '+json.dumps(result),flush=True)
        return result
    except BaseException as exc:
        write(run/'status.json',{'status':'failed','error':repr(exc),'seconds':time.time()-start})
        raise


ARM_WEIGHTS={
    'B0':{'coarse':0.,'fine':0.,'rotated_bce':0.,'exist_pair':0.},
    'S1_saliency_only':{'coarse':1.,'fine':1.,'rotated_bce':0.,'exist_pair':0.},
}


def verify_freeze(split):
    freeze=json.loads((ROOT/'configs/SALIENCY_V5_FREEZE.json').read_text())
    for path,expected in freeze['file_sha256'].items():
        if digest(path)!=expected: raise RuntimeError(f'Freeze mismatch: {path}')
    if freeze['arms']!=ARM_WEIGHTS: raise RuntimeError('Arm definition differs from freeze')


def gradient_probe(model,components,weights):
    params=[p for n,p in model.named_parameters() if p.requires_grad and
            n.startswith(('transformer.','input_vid_proj.','input_txt_proj.'))]
    def grad(value):
        return [torch.zeros_like(p) if g is None else g.detach() for p,g in
                zip(params,torch.autograd.grad(value,params,retain_graph=True,allow_unused=True))]
    base=grad(components['gmr']);base_sq=sum(x.square().sum() for x in base)
    total=[torch.zeros_like(p) for p in params];result={}
    def measure(values):
        norm_sq=sum(x.square().sum() for x in values)
        dot=sum((x*y).sum() for x,y in zip(base,values))
        return {'norm':float(norm_sq.sqrt()),'norm_ratio_to_gmr':float((norm_sq/base_sq.clamp_min(1e-20)).sqrt()),
                'cosine_to_gmr':float(dot/(base_sq*norm_sq).clamp_min(1e-20).sqrt())}
    for key,value in components.items():
        if key=='gmr': continue
        values=grad(value);result[key]={'loss':float(value.detach()),**measure(values),'active':bool(weights[key])}
        if weights[key]:
            for dst,src in zip(total,values): dst.add_(src,alpha=weights[key])
    return {'scope':'first shuffled training batch; train mode; pre-clipping; shared transformer/input projections',
            'gmr_norm':float(base_sq.sqrt()),'components':result,'active_auxiliary':measure(total),
            'limitation':'local raw gradient geometry, not AdamW update attribution'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',required=True,choices=['A1','C1'])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--arms',nargs='+',choices=list(ARM_WEIGHTS),default=list(ARM_WEIGHTS))
    parser.add_argument('--run-id',default='saliency_v5_50ep')
    args=parser.parse_args()
    if args.epochs!=50 or args.lr!=1e-5 or args.run_id!='saliency_v5_50ep':
        raise RuntimeError('Arguments differ from frozen saliency v5 design')
    verify_freeze(args.fold)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    source=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2')/args.fold/'moment/best.ckpt'
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    opt=copy.deepcopy(checkpoint['opt']);opt.device=args.device;opt.mr_only=True;opt.lw_saliency=0
    opt.n_epoch=args.epochs;opt.lr=args.lr;opt.seed=3407
    if opt.bsz!=16: raise RuntimeError('Frozen batch size must be 16')
    opt.train_path=str(ROOT/'data/formal'/args.fold/'train.jsonl');opt.eval_path=str(ROOT/'data/formal'/args.fold/'seen_val.jsonl')
    datasets={role:CachedDataset(opt,args.fold,role) for role in ['train','seen_val']}
    trainvid={r['vid'] for r in datasets['train'].rows}
    if any(trainvid & {r['vid'] for r in datasets[role].rows} for role in ['seen_val']):
        raise RuntimeError('Video split overlap')
    write(ROOT/'records/saliency_v5'/('IMPORT_AUDIT_'+args.fold+'_'+'_'.join(args.arms)+'.json'),{'models':sys.modules[build_model.__module__].__file__,
        'dataset':sys.modules[StartEndDataset.__module__].__file__,'checkpoint_source':str(source),'checkpoint_sha256':digest(source),
        'train_chain_support':len(datasets['train'].hn),'device':args.device,'python':sys.version,'torch':torch.__version__})
    print(f'Loaded datasets {args.fold}',flush=True)
    results=[]
    for arm in args.arms:
        torch.cuda.reset_peak_memory_stats()
        results.append(run_arm(args,opt,checkpoint,datasets,arm))
    write(ROOT/'records/saliency_v5'/('RESULTS_'+args.fold+'_'+args.run_id+'_'+'_'.join(args.arms)+'.json'),results)


if __name__=='__main__':main()
