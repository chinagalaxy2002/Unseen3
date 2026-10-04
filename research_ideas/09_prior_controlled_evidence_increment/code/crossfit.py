"""Query-only OOF training; never uses Seen-val/Novel labels for selection."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from core import auc, digest, labels, write_json


class Cq(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def video_folds(rows, seed, count):
    videos = np.asarray(sorted({str(r['vid']) for r in rows}))
    if len(videos) < count:
        raise ValueError('Too few videos for crossfit')
    rng = np.random.default_rng(seed)
    assignment = {v: i % count for i, v in enumerate(rng.permutation(videos))}
    return np.asarray([assignment[str(r['vid'])] for r in rows]), assignment


def heldin_split(rows, outer, excluded, seed, fraction):
    allowed = sorted({str(r['vid']) for i, r in enumerate(rows) if outer[i] != excluded})
    order = np.random.default_rng(seed).permutation(allowed)
    nval = max(1, int(np.ceil(len(order) * fraction)))
    if nval >= len(order):
        raise ValueError('Insufficient held-in videos')
    validation = set(order[:nval])
    fit = np.asarray([i for i, r in enumerate(rows) if outer[i] != excluded and str(r['vid']) not in validation])
    val = np.asarray([i for i, r in enumerate(rows) if outer[i] != excluded and str(r['vid']) in validation])
    return fit, val


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    canonical = np.concatenate([model(torch.as_tensor(unique[start:start + 256], device=device)).cpu().numpy()
                               for start in range(0, len(unique), 256)]).astype(np.float64)
    return canonical[inverse]


def train_crossfit(rows, means, directory, freeze, freeze_hash, device):
    """Receives train rows/features only; saves selected model then OOF scores."""
    directory = Path(directory)
    outer, assignment = video_folds(rows, freeze['seed'], freeze['crossfit_video_folds'])
    assignment_path = directory / 'outer_video_assignment.json'
    if assignment_path.exists():
        if json.loads(assignment_path.read_text()) != assignment:
            raise ValueError('Crossfit assignment changed')
    else:
        write_json(assignment_path, assignment)
    y, oof = labels(rows), []
    for seed in freeze['crossfit_seeds']:
        scores = np.empty(len(rows), dtype=float)
        for k in range(freeze['crossfit_video_folds']):
            derived = seed + 1009 * k
            fit, val = heldin_split(rows, outer, k, derived, freeze['stage2']['heldin_validation_fraction'])
            if set(y[fit]) != {0, 1} or set(y[val]) != {0, 1}:
                raise ValueError(f'Frozen held-in partition lacks both classes: seed={seed}, fold={k}')
            excluded = np.flatnonzero(outer == k)
            model_path = directory / f'cq_seed{seed}_outer{k}.pt'
            if model_path.with_suffix('.json').exists():
                metadata = json.loads(model_path.with_suffix('.json').read_text())
                if metadata['freeze_sha256'] != freeze_hash or digest(model_path) != metadata['model_sha256']:
                    raise ValueError('Existing Cq artifact changed')
                if (metadata['fit_indices'] != fit.tolist() or metadata['validation_indices'] != val.tolist() or
                    metadata['oof_indices'] != excluded.tolist()):
                    raise ValueError('Existing Cq partition changed')
                model = Cq().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                scores[excluded] = predict(model, means[excluded], device)
                print(f'Cq seed={seed} outer={k}: resumed verified model', flush=True)
                continue
            torch.manual_seed(derived)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(derived)
            model = Cq().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            history, best, best_auc, best_epoch = [], None, -np.inf, None
            for epoch in range(1, freeze['stage2']['cq_epochs'] + 1):
                model.train()
                order = np.random.default_rng(derived + epoch).permutation(fit)
                for start in range(0, len(order), 32):
                    ix = order[start:start + 32]
                    logits = model(torch.as_tensor(means[ix], device=device))
                    loss = nn.functional.binary_cross_entropy_with_logits(logits,
                        torch.as_tensor(y[ix], dtype=torch.float32, device=device))
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite Cq loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), 1)
                    if not torch.isfinite(norm):
                        raise RuntimeError('Nonfinite Cq gradient')
                    optimizer.step()
                score = auc(y[val], predict(model, means[val], device))
                history.append({'epoch': epoch, 'heldin_val_auc': score})
                if score > best_auc + 1e-15:
                    best_auc, best_epoch = score, epoch
                    best = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            model.load_state_dict(best)
            scores[excluded] = predict(model, means[excluded], device)
            model_path = directory / f'cq_seed{seed}_outer{k}.pt'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best, model_path)
            write_json(model_path.with_suffix('.json'), {'freeze_sha256': freeze_hash, 'model_sha256': digest(model_path),
                'seed': seed, 'derived_seed': derived, 'outer_fold': k, 'fit_indices': fit.tolist(),
                'validation_indices': val.tolist(), 'oof_indices': excluded.tolist(),
                'best_epoch': best_epoch, 'best_heldin_val_auc': best_auc, 'history': history,
                'novel_labels_used': False, 'seen_val_labels_used': False})
            print(f'Cq seed={seed} outer={k} selected={best_epoch} AUC={best_auc:.5f}', flush=True)
        oof.append(scores)
    oof = np.stack(oof)
    np.save(directory / 'oof_by_seed.npy', oof)
    return oof.mean(0)


def predict_crossfit(means, directory, freeze, freeze_hash, device):
    import json
    all_scores = []
    for seed in freeze['crossfit_seeds']:
        per_seed = []
        for k in range(freeze['crossfit_video_folds']):
            path = Path(directory) / f'cq_seed{seed}_outer{k}.pt'
            metadata = json.loads(path.with_suffix('.json').read_text())
            if metadata['freeze_sha256'] != freeze_hash or metadata['model_sha256'] != digest(path):
                raise ValueError('Crossfit artifact changed')
            model = Cq().to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            per_seed.append(predict(model, means, device))
        all_scores.append(np.mean(per_seed, axis=0))
    return np.mean(all_scores, axis=0)
