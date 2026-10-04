"""Train-only, initial-checkpoint gradient probes. No optimizer or model selection."""
import copy,json,random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_formal import (ROOT,CachedDataset,build_model,start_end_collate,prepare_batch_inputs,
                         gt_mask,pad_hn,coarse,fine,write,digest)
BASE='/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2'

def stats(a,b):
    dot=sum((x*y).sum() for x,y in zip(a,b));aa=sum(x.square().sum() for x in a);bb=sum(x.square().sum() for x in b)
    return {'norm':float(bb.sqrt()),'norm_ratio_to_gmr':float((bb/aa.clamp_min(1e-20)).sqrt()),
            'cosine_to_gmr':float(dot/(aa*bb).clamp_min(1e-20).sqrt())}

def run():
    torch.set_num_threads(4);results={}
    for split in ['A1','C1']:
        random.seed(3407);np.random.seed(3407);torch.manual_seed(3407)
        ck=torch.load(f'{BASE}/{split}/moment/best.ckpt',map_location='cpu',weights_only=False)
        opt=copy.deepcopy(ck['opt']);opt.device='cuda:0';opt.mr_only=True;opt.lw_saliency=0
        ds=CachedDataset(opt,split,'train');model,criterion=build_model(opt)
        model.load_state_dict(ck['model']);model.to(opt.device).eval();criterion.to(opt.device).eval()
        params=[p for name,p in model.named_parameters() if p.requires_grad and
                (name.startswith('transformer.') or name.startswith('input_vid_proj.') or name.startswith('input_txt_proj.'))]
        loader=DataLoader(ds,batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(3407),collate_fn=start_end_collate)
        results[split]=[]
        for index,(metas,batch) in enumerate(loader):
            if index>=3:break
            inputs,targets=prepare_batch_inputs(batch,opt.device);o=model(**inputs)
            original=criterion(o,targets);gmr=sum(v*criterion.weight_dict[k] for k,v in original.items() if k in criterion.weight_dict)
            rotated=dict(inputs);rotated['src_txt']=inputs['src_txt'].roll(-1,0);rotated['src_txt_mask']=inputs['src_txt_mask'].roll(-1,0)
            n=model(**rotated);valid=inputs['src_vid_mask'];gt=gt_mask(metas,valid,opt.clip_length);pos=targets['exist_label'].bool()
            parts={'coarse':coarse(o['saliency_scores'],n['saliency_scores'],gt,valid,pos),
                   'rotated_bce':F.binary_cross_entropy_with_logits(n['pred_exist_logits'],torch.zeros_like(n['pred_exist_logits'])),
                   'weighted_exist_pair':.2*F.relu(.2-o['pred_exist_logits'][pos]+n['pred_exist_logits'][pos]).mean()}
            ids=[i for i,m in enumerate(metas) if pos[i] and str(m['qid']) in ds.hn][:4]
            if ids:
                text,mask=pad_hn(ds,metas,ids,opt.device)
                hard=model(src_txt=text,src_txt_mask=mask,src_vid=inputs['src_vid'][ids].repeat(3,1,1),src_vid_mask=valid[ids].repeat(3,1))
                parts['fine']=fine(o['saliency_scores'][ids],hard['saliency_scores'].reshape(3,len(ids),-1),n['saliency_scores'][ids],gt[ids],valid[ids])
            def grads(loss):
                return [torch.zeros_like(p) if g is None else g for p,g in zip(params,torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True))]
            g=grads(gmr);combined=[torch.zeros_like(p) for p in params];row={'batch':index,'gmr_loss':float(gmr),'components':{}}
            for name,loss in parts.items():
                b=grads(loss);row['components'][name]={'loss':float(loss),**stats(g,b)}
                for acc,v in zip(combined,b):acc.add_(v)
                del b
            row['combined_auxiliary']=stats(g,combined);results[split].append(row)
            print(split,index,json.dumps(row),flush=True)
            del g,combined,o,n,parts,gmr
        del model,criterion,ds;torch.cuda.empty_cache()
    write(ROOT/'records/formal/GRADIENT_DIAGNOSTICS.json',{'scope':'three train minibatches at each canonical checkpoint',
        'mode':'eval (dropout disabled), autograd enabled, no optimizer updates','parameters':'shared transformer and input projections',
        'limitations':'local gradient geometry; not a causal ablation or full-training estimate',
        'script_sha256':digest(__file__),'results':results})
if __name__=='__main__':run()
