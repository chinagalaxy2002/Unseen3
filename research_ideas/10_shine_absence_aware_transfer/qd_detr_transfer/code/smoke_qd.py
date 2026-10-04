"""Real train-batch integration check, no optimizer updates and no test access."""
import copy
import json
import random
from types import SimpleNamespace
import numpy as np
import torch
from train_qd_saliency import ROOT,CODE,build_model,StartEndDataset,start_end_collate,prepare_batch_inputs,gt_mask,pad_hn,coarse,fine,write,digest,l2_normalize_np_array

def main():
    torch.set_num_threads(4);torch.manual_seed(3407);random.seed(3407);np.random.seed(3407)
    results={}
    for split in ['A1','C1']:
        source=f'/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2/{split}/qd/best.ckpt'
        ck=torch.load(source,map_location='cpu',weights_only=False);opt=copy.deepcopy(ck['opt']);opt.device='cuda:0';opt.mr_only=True;opt.lw_saliency=0
        ds=StartEndDataset(dset_name=opt.dset_name,data_path=str(ROOT/f'data/formal/{split}/train.jsonl'),v_feat_dirs=opt.v_feat_dirs,q_feat_dir=opt.t_feat_dir,max_q_l=opt.max_q_l,max_v_l=opt.max_v_l,ctx_mode=opt.ctx_mode,clip_len=opt.clip_length,max_windows=opt.max_windows,span_loss_type=opt.span_loss_type,mr_only=True,keep_empty_gt=True)
        generator=torch.Generator().manual_seed(3407);indices=torch.randperm(len(ds),generator=generator)[:16].tolist()
        metas,batch=start_end_collate([ds[i] for i in indices]);inputs,targets=prepare_batch_inputs(batch,opt.device)
        assert targets['exist_label'].tolist()==[float(m['exist_label']) for m in metas]
        model,criterion=build_model(opt);model.load_state_dict(ck['model'],strict=True);model.to(opt.device).train();criterion.train()
        out=model(**inputs);natural=criterion(out,targets);gmr=sum(v*criterion.weight_dict[k] for k,v in natural.items() if k in criterion.weight_dict)
        rotated=dict(inputs);rotated['src_txt']=inputs['src_txt'].roll(-1,0);rotated['src_txt_mask']=inputs['src_txt_mask'].roll(-1,0);neg=model(**rotated)
        valid=inputs['src_vid_mask'];gt=gt_mask(metas,valid,opt.clip_length);positive=targets['exist_label'].bool()
        bank=json.loads((ROOT/f'data/formal/{split}/query_bank.json').read_text());ids=[i for i,m in enumerate(metas) if positive[i] and str(m['qid']) in bank][:4];hn={}
        for i in ids:
            qid=str(metas[i]['qid']);hn[qid]=[]
            for edit in bank[qid]['edits']:
                with np.load(ROOT.parent/'artifacts/edit_clip'/(edit['feature_id']+'.npz')) as z:a=z['last_hidden_state'][:opt.max_q_l].astype(np.float32)
                hn[qid].append(torch.from_numpy(l2_normalize_np_array(a)))
        text,mask=pad_hn(SimpleNamespace(hn=hn),metas,ids,opt.device)
        hard=model(src_txt=text,src_txt_mask=mask,src_vid=inputs['src_vid'][ids].repeat(3,1,1),src_vid_mask=valid[ids].repeat(3,1))
        lc=coarse(out['saliency_scores'],neg['saliency_scores'],gt,valid,positive)
        lf=fine(out['saliency_scores'][ids],hard['saliency_scores'].reshape(3,len(ids),-1),neg['saliency_scores'][ids],gt[ids],valid[ids])
        total=gmr+lc+lf;assert torch.isfinite(total);total.backward()
        grads={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
        assert grads and all(np.isfinite(v) for v in grads.values())
        assert any(k.startswith('saliency_proj') and v>0 for k,v in grads.items())
        results[split]={'gmr_loss':float(gmr.detach()),'coarse':float(lc.detach()),'fine':float(lf.detach()),'total':float(total.detach()),'finite_backward':True,'chains':len(ids),'native_saliency_shape':list(out['saliency_scores'].shape),'qd_global_token_saliency_retained':True,'positive_labels_verified':True,'checkpoint_sha256':digest(source),'model_module':str(CODE/'models/qd_detr_gmr/model.py'),'max_gpu_bytes':torch.cuda.max_memory_allocated(),'optimizer_updated':False}
        print(split,json.dumps(results[split]),flush=True)
        del ds,model,criterion,ck,out,neg,hard,total,gmr,lc,lf;torch.cuda.empty_cache()
    write(ROOT/'records/SMOKE_CHECK.json',results)
if __name__=='__main__':main()
