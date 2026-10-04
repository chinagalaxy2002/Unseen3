"""Check mathematical parity, masking and real GMR gradient flow."""
import importlib.util
import json
from pathlib import Path
import torch
from shine_losses import coarse,fine
from train_pilot import ROOT, PROJECT, CachedDataset, gt_mask, pad_hn, build_model, start_end_collate, prepare_batch_inputs


def main():
    torch.set_num_threads(4)
    source=ROOT/'vendor/shine/shine/ctf_ranking.py'
    spec=importlib.util.spec_from_file_location('upstream_ctf',source)
    up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
    torch.manual_seed(3407)
    n,t=3,11
    pos=torch.randn(n,t,requires_grad=True);neg=torch.randn(n,t,requires_grad=True)
    hn=torch.randn(3,n,t,requires_grad=True)
    valid=torch.ones(n,t);gt=torch.zeros(n,t);gt[:,2:6]=1
    outputs={'saliency_scores':pos,'saliency_scores_neg':neg}
    outputs.update({f'saliency_scores_hn{i+1}':hn[i] for i in range(3)})
    targets={'saliency_all_labels':gt,'video_length':torch.full((n,),t)}
    c=coarse(pos,neg,gt,valid,torch.ones(n,dtype=torch.bool))
    f=fine(pos,hn,neg,gt,valid)
    assert torch.allclose(c,up.coarse_ranking(targets,outputs,1.,2.),atol=1e-6)
    assert torch.allclose(f,up.fine_ranking(targets,outputs,rt='red'),atol=1e-6)
    valid[:,-2:]=0;gt[:,-2:]=0
    loss=coarse(pos,neg,gt,valid,torch.ones(n,dtype=torch.bool))+fine(pos,hn,neg,gt,valid)
    loss.backward()
    assert (pos.grad[:,-2:]==0).all() and (hn.grad[:,:,-2:]==0).all()
    empty=coarse(pos,neg,torch.zeros_like(gt),valid,torch.zeros(n,dtype=torch.bool))
    assert empty==0 and torch.isfinite(empty)
    ckpt=torch.load(PROJECT/'results/moment_detr_gmr_evidence_v5/inner_baselines/A1_action_01/best.ckpt',map_location='cpu',weights_only=False)
    opt=ckpt['opt'];opt.device='cuda:0'
    model,criterion=build_model(opt);model.load_state_dict(ckpt['model']);model.to(opt.device);criterion.to(opt.device)
    ds=CachedDataset(opt,'A1_action_01','inner_train')
    idx=[i for i,r in enumerate(ds.rows) if str(r['qid']) in ds.hn][:2]
    idx.append(next(i for i,r in enumerate(ds.rows) if not r['exist_label']))
    metas,batch=start_end_collate([ds[i] for i in idx])
    inputs,targets=prepare_batch_inputs(batch,opt.device)
    out=model(**inputs)
    neg_inputs=dict(inputs);neg_inputs['src_txt']=inputs['src_txt'].roll(-1,0);neg_inputs['src_txt_mask']=inputs['src_txt_mask'].roll(-1,0)
    negout=model(**neg_inputs)
    text,mask=pad_hn(ds,metas,[0,1],opt.device)
    hard=model(src_txt=text,src_txt_mask=mask,src_vid=inputs['src_vid'][:2].repeat(3,1,1),src_vid_mask=inputs['src_vid_mask'][:2].repeat(3,1))
    gt=gt_mask(metas,inputs['src_vid_mask'],opt.clip_length)
    lc=coarse(out['saliency_scores'],negout['saliency_scores'],gt,inputs['src_vid_mask'],targets['exist_label'].bool())
    lf=fine(out['saliency_scores'][:2],hard['saliency_scores'].reshape(3,2,-1),negout['saliency_scores'][:2],gt[:2],inputs['src_vid_mask'][:2])
    import torch.nn.functional as F
    # Check temporal-only gradients before including base GMR losses.
    (lc+lf).backward(retain_graph=True)
    fusion=sum(float(p.grad.abs().sum()) for p in model.transformer.parameters() if p.grad is not None)
    assert fusion>0 and model.saliency_proj.weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    base=criterion(out,targets)
    loss=sum(v*criterion.weight_dict[k] for k,v in base.items() if k in criterion.weight_dict)+lc+lf+F.binary_cross_entropy_with_logits(negout['pred_exist_logits'],torch.zeros(3,device=opt.device))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert sum(float(p.grad.abs().sum()) for p in model.exist_head.parameters() if p.grad is not None)>0
    report={'status':'passed','upstream_coarse_parity':True,'upstream_red_fine_parity':True,'padding_zero_gradient':True,
            'empty_null_safe':True,'real_mixed_batch_backward':True,'temporal_fusion_gradient_l1':fusion,
            'existence_head_gradient':True,'formal_U_accessed':False}
    (ROOT/'records/IMPLEMENTATION_VALIDATION.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':main()
