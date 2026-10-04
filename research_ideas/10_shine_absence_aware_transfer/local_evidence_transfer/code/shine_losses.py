"""SHINE coarse/fine transfer with padding/null-set support.

Fine distance and red ranking follow vendor/shine/shine/ctf_ranking.py.
Batch-rotated absence and the hierarchy are user-authorized assumptions.
"""
import torch
import torch.nn.functional as F


def coarse(pos, neg, gt, valid, positive, margins=(1.,2.), q=8):
    intra, inter = [], []
    for i in positive.nonzero(as_tuple=True)[0].tolist():
        inside = gt[i].bool() & valid[i].bool()
        outside = ~gt[i].bool() & valid[i].bool()
        if not inside.any():
            continue
        k=max(int(inside.sum())//q,1)
        top=pos[i,inside].topk(k).values.mean()
        inter.append(F.relu(margins[1]+neg[i,inside].topk(k).values.mean()-top))
        if outside.any():
            intra.append(F.relu(margins[0]+pos[i,outside].max()-top))
    zero=pos.sum()*0
    return (torch.stack(intra).mean() if intra else zero)+(torch.stack(inter).mean() if inter else zero)


def div_loss(target, logits, valid, gt=False, epsilon=1e-9):
    label=target if gt else target.sigmoid()
    # Match upstream epsilon rather than substitute complete BCE or KL.
    values=-label.detach()*torch.log(logits.sigmoid()+epsilon)
    return (values*valid).sum(-1)/valid.sum(-1).clamp_min(1)


def fine(pos, hn, neg, gt, valid, margin=.25):
    if hn.shape[0]!=3:
        raise ValueError('SHINE red ranking requires three levels')
    distances=[div_loss(gt,pos,valid,gt=True)]
    distances.extend(div_loss(pos,hn[level],valid) for level in range(3))
    distances.append(div_loss(pos,neg,valid))
    return sum(F.relu(margin+a-b).mean() for a,b in zip(distances[:-1],distances[1:]))
