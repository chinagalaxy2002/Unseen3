"""Conditional ranking surrogates. These do not directly differentiate AUROC."""
import torch
from torch import nn
from torch.nn import functional as F


class SemanticRobustAUCLoss(nn.Module):
    def __init__(self, margin=1.0, tau=0.1):
        super().__init__()
        if tau <= 0: raise ValueError('tau must be positive')
        self.margin, self.tau = float(margin), float(tau)

    def conditional(self, scores, labels, groups):
        losses=[]
        for group in torch.unique(groups):
            pos=scores[(groups==group)&(labels>0.5)]
            neg=scores[(groups==group)&(labels<=0.5)]
            if pos.numel() and neg.numel():
                losses.append(F.softplus(self.margin-pos[:,None]+neg[None,:]).mean())
        if not losses: return scores.sum()*0.0, 0
        # Exactly the specified smooth max, without normalization by group count.
        return self.tau*torch.logsumexp(torch.stack(losses)/self.tau,dim=0), len(losses)

    def forward(self, scores, targets):
        scores=scores.reshape(-1); labels=targets['exist_label'].reshape(-1)
        action, na=self.conditional(scores,labels,targets['group_action'])
        comp, nc=self.conditional(scores,labels,targets['group_composition'])
        matched=[]; exact_count=comp_count=action_count=0
        for p in torch.where(labels>0.5)[0]:
            neg=labels<=0.5
            same_query=neg & (targets['query_group']==targets['query_group'][p])
            same_comp=neg & (targets['group_composition']==targets['group_composition'][p])
            same_action=neg & (targets['group_action']==targets['group_action'][p])
            if same_query.any(): chosen=same_query; exact_count+=int(chosen.sum())
            elif same_comp.any(): chosen=same_comp; comp_count+=int(chosen.sum())
            elif same_action.any(): chosen=same_action; action_count+=int(chosen.sum())
            else: continue
            matched.append(F.softplus(self.margin-scores[p]+scores[chosen]).mean())
        zero=scores.sum()*0.0
        scalar=lambda x: scores.new_tensor(float(x))
        return {'loss_robust_auc':0.5*action+0.5*comp,
                'loss_same_semantic':torch.stack(matched).mean() if matched else zero,
                'robust_action_surrogate':action.detach(), 'robust_composition_surrogate':comp.detach(),
                'auc_valid_action_groups':scalar(na), 'auc_valid_composition_groups':scalar(nc),
                'matched_exact_query_pairs':scalar(exact_count), 'matched_composition_pairs':scalar(comp_count),
                'matched_action_pairs':scalar(action_count)}
