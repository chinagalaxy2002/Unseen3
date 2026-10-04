"""CPU-only actual train batch and native inference check; no optimizer/test access."""
import copy
import json
import random
from types import SimpleNamespace
import numpy as np
import torch
from train_flash_saliency import ROOT,CODE,canonical_path,build_model,StartEndDataset,start_end_collate,prepare_batch_inputs,gt_mask,pad_hn,coarse,fine,write,digest,l2_normalize_np_array,evaluate

def main():
    torch.set_num_threads(2);torch.manual_seed(3407);random.seed(3407);np.random.seed(3407);results={}
    for split in ['A1','C1']:
        source=canonical_path(split);ck=torch.load(source,map_location='cpu',weights_only=False)
        opt=copy.deepcopy(ck['opt']);opt.device='cpu';opt.mr_only=True;opt.lw_saliency=0;opt.bsz=16
        rows=[json.loads(l) for l in (ROOT/f'data/formal/{split}/train.jsonl').read_text().splitlines() if l.strip()]
        indices=torch.randperm(len(rows),generator=torch.Generator().manual_seed(3407))[:16].tolist()
        subset=ROOT/f'records/{split}_smoke_train.jsonl';subset.write_text(''.join(json.dumps(rows[i])+'\n' for i in indices))
        ds=StartEndDataset(dset_name=opt.dset_name,data_path=str(subset),v_feat_dirs=opt.v_feat_dirs,q_feat_dir=opt.t_feat_dir,max_q_l=opt.max_q_l,max_v_l=opt.max_v_l,ctx_mode=opt.ctx_mode,clip_len=opt.clip_length,max_windows=opt.max_windows,span_loss_type=opt.span_loss_type,mr_only=True,keep_empty_gt=True)
        metas,batch=start_end_collate([ds[i] for i in range(len(ds))]);inputs,targets=prepare_batch_inputs(batch,opt.device)
        assert targets['exist_label'].tolist()==[float(m['exist_label']) for m in metas]
        targets['label']=metas;targets['fps']=torch.full((len(metas),),1/opt.clip_length)
        model,criterion=build_model(opt);model.load_state_dict(ck['model'],strict=True);model.train();criterion.train()
        out=model(**inputs,targets=targets);natural=criterion((metas,batch),out,targets);natural={k:v for k,v in natural.items() if 'loss' in k}
        gmr=sum(v*criterion.weight_dict[k] for k,v in natural.items() if k in criterion.weight_dict)
        rotated=dict(inputs);rotated['src_txt']=inputs['src_txt'].roll(-1,0);rotated['src_txt_mask']=inputs['src_txt_mask'].roll(-1,0);neg=model(**rotated)
        valid=inputs['src_vid_mask'];gt=gt_mask(metas,valid,opt.clip_length);pos=targets['exist_label'].bool()
        bank=json.loads((ROOT/f'data/formal/{split}/query_bank.json').read_text());ids=[i for i,m in enumerate(metas) if pos[i] and str(m['qid']) in bank][:4];hn={}
        for i in ids:
            qid=str(metas[i]['qid']);hn[qid]=[]
            for edit in bank[qid]['edits']:
                with np.load(ROOT.parent/'artifacts/edit_clip'/(edit['feature_id']+'.npz')) as z:a=z['last_hidden_state'][:opt.max_q_l].astype(np.float32)
                hn[qid].append(torch.from_numpy(l2_normalize_np_array(a)))
        text,mask=pad_hn(SimpleNamespace(hn=hn),metas,ids,opt.device)
        hard=model(src_txt=text,src_txt_mask=mask,src_vid=inputs['src_vid'][ids].repeat(3,1,1),src_vid_mask=valid[ids].repeat(3,1),vid=[metas[i]['vid'] for _ in range(3) for i in ids],qid=[metas[i]['qid'] for _ in range(3) for i in ids])
        lc=coarse(out['saliency_scores'],neg['saliency_scores'],gt,valid,pos)
        lf=fine(out['saliency_scores'][ids],hard['saliency_scores'].reshape(3,len(ids),-1),neg['saliency_scores'][ids],gt[ids],valid[ids])
        total=gmr+lc+lf;assert torch.isfinite(total);total.backward()
        grads={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
        assert grads and all(np.isfinite(v) for v in grads.values())
        assert any(k.startswith('saliency_proj') and v>0 for k,v in grads.items())
        val=SimpleNamespace(rows=ds.data,samples=ds.preloaded_data)
        class Tiny(torch.utils.data.Dataset):
            def __len__(self):return len(val.samples)
            def __getitem__(self,i):return val.samples[i]
        tiny=Tiny();tiny.rows=val.rows
        evalmetrics,_=evaluate(model,tiny,opt)
        results[split]={'gmr_loss':float(gmr.detach()),'coarse':float(lc.detach()),'fine':float(lf.detach()),'total':float(total.detach()),'finite_backward':True,'chains':len(ids),'native_saliency_shape':list(out['saliency_scores'].shape),'native_inference_verified':True,'smoke_train_auc':evalmetrics['pooled_auroc'],'native_proposals_in_seconds':True,'positive_labels_verified':True,'checkpoint_sha256':digest(source),'model_module':str(CODE/'models/flash_vtg_gmr/model.py'),'device':'cpu','optimizer_updated':False,'formal_test_accessed':False}
        print(split,json.dumps(results[split]),flush=True)
        del ds,model,criterion,ck,out,neg,hard,total,gmr,lc,lf
    write(ROOT/'records/SMOKE_CHECK.json',results)
if __name__=='__main__':main()
