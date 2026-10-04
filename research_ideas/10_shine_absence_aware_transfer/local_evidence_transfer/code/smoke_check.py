"""Actual train-only finite/path checks, no optimizer or formal-test access."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_local_evidence import ROOT,DATA_ROOT,build_model,StartEndDataset,start_end_collate,prepare_batch_inputs,write
from evidence_bridge import attach_local_evidence,pooled_evidence
BASE=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2')

def main():
 torch.set_num_threads(4);records={}
 for split in ['A1','C1']:
  torch.manual_seed(3407);ck=torch.load(BASE/split/'moment/best.ckpt',map_location='cpu',weights_only=False);opt=copy.deepcopy(ck['opt']);opt.device='cpu';opt.mr_only=True;opt.lw_saliency=0
  ds=StartEndDataset(dset_name=opt.dset_name,domain=None,data_path=str(DATA_ROOT/f'data/formal/{split}/train.jsonl'),v_feat_dirs=opt.v_feat_dirs,q_feat_dir=opt.t_feat_dir,v_feat_types=opt.v_feat_types,max_q_l=opt.max_q_l,max_v_l=opt.max_v_l,ctx_mode=opt.ctx_mode,clip_len=opt.clip_length,max_windows=opt.max_windows,span_loss_type=opt.span_loss_type,mr_only=True,keep_empty_gt=True)
  _,batch=next(iter(DataLoader([ds[0],ds[1]],batch_size=2,collate_fn=start_end_collate)))
  inputs,targets=prepare_batch_inputs(batch,'cpu');model,criterion=build_model(opt);model.load_state_dict(ck['model'],strict=True);model.eval()
  with torch.no_grad():base=model(**inputs)['pred_exist_logits'].clone()
  model=attach_local_evidence(model,'saliency');out=model(**inputs)
  assert torch.equal(out['pred_exist_logits'],base)
  assert torch.allclose(out['local_evidence_weights'].sum(1),torch.ones(2))
  assert (out['local_evidence_weights'][~inputs['src_vid_mask'].bool()]==0).all()
  losses=criterion(out,targets);loss=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
  loss.backward();initial_grad=float(model.local_evidence_head[-1].weight.grad.norm());assert initial_grad>0 and torch.isfinite(loss)
  model.zero_grad(set_to_none=True)
  with torch.no_grad():model.local_evidence_head[-1].weight.fill_(.001)
  out=model(**inputs);sal_grad=torch.autograd.grad(out['local_evidence_logit'].sum(),out['saliency_scores'],retain_graph=True)[0]
  assert torch.isfinite(sal_grad).all() and sal_grad.abs().sum()>0
  padded_mem=torch.randn(2,4,8);padded_sal=torch.randn(2,4,requires_grad=True);valid=torch.tensor([[1,1,0,0],[1,1,1,0]])
  pool,w,_=pooled_evidence(padded_mem,padded_sal,valid,'saliency');assert (w[~valid.bool()]==0).all()
  altered=padded_mem.clone();altered[~valid.bool()]=1e9
  assert torch.allclose(pool,pooled_evidence(altered,padded_sal,valid,'saliency')[0])
  uni=pooled_evidence(padded_mem,padded_sal,valid,'uniform')[0]
  assert torch.equal(uni,pooled_evidence(padded_mem,padded_sal+torch.randn_like(padded_sal),valid,'uniform')[0])
  records[split]={'zero_init_replays_canonical_exactly':True,'natural_loss_finite':True,'initial_local_final_layer_gradient_norm':initial_grad,'local_residual_to_saliency_gradient_norm':float(sal_grad.norm()),'padding_invariant':True,'uniform_control_independent_of_saliency':True,'data':'actual first2 train rows','optimizer_updates':0}
  print(split,json.dumps(records[split]),flush=True)
 write(ROOT/'records/SMOKE_CHECK.json',{'formal_test_accessed':False,'records':records})
if __name__=='__main__':main()
