import torch

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if output.numel() == 0:
        return [torch.tensor(0.0, device=output.device)]

    if isinstance(target, int):
        target = torch.full((output.shape[0],), target, dtype=torch.int64, device=output.device)

    if target.numel() == 0:
        return [torch.tensor(0.0, device=output.device)]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
