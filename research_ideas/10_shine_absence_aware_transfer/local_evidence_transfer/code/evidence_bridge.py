"""Differentiable saliency-to-existence residual; no GT or bank at inference."""
from types import MethodType
import torch
from torch import nn

def pooled_evidence(memory,saliency,valid,mode):
    valid=valid.bool()
    if not valid.any(dim=1).all():raise RuntimeError('Empty valid video sequence')
    if mode=='saliency':
        weights=torch.softmax(saliency.masked_fill(~valid,float('-inf')),dim=1)
    elif mode=='uniform':
        weights=valid.to(memory.dtype)/valid.sum(1,keepdim=True)
    else:raise ValueError(mode)
    pooled=(weights.unsqueeze(-1)*memory).sum(1)
    entropy=-(weights*weights.clamp_min(1e-12).log()).sum(1)
    return pooled,weights,entropy

def attach_local_evidence(model,mode):
    # Preserve parent's post-model-construction RNG for matched dropout exposure.
    dim=model.saliency_proj.in_features
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3407)
        model.local_evidence_head=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,1))
        nn.init.zeros_(model.local_evidence_head[-1].weight)
        nn.init.zeros_(model.local_evidence_head[-1].bias)
    model.local_evidence_pool=mode
    original=model.forward
    def forward(this,src_txt,src_txt_mask,src_vid,src_vid_mask,src_aud=None,src_aud_mask=None):
        out=original(src_txt,src_txt_mask,src_vid,src_vid_mask,src_aud,src_aud_mask)
        pooled,weights,entropy=pooled_evidence(out['video_memory'],out['saliency_scores'],src_vid_mask,this.local_evidence_pool)
        residual=this.local_evidence_head(pooled).squeeze(-1)
        out['base_exist_logits']=out['pred_exist_logits']
        out['local_evidence_logit']=residual
        out['local_evidence_weights']=weights
        out['local_evidence_entropy']=entropy
        out['pred_exist_logits']=out['pred_exist_logits']+residual
        return out
    model.forward=MethodType(forward,model)
    return model
