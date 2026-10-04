"""Collect completed paired, single-seed runs; never select on Novel-dev."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def interval(base,method):
    if [(r['qid'],r['vid'],r['exist_label']) for r in base] != [(r['qid'],r['vid'],r['exist_label']) for r in method]:
        raise RuntimeError('Prediction identity mismatch')
    y=np.array([r['exist_label'] for r in base])
    b=np.array([r['logit'] for r in base]);m=np.array([r['logit'] for r in method])
    _,inverse=np.unique([r['vid'] for r in base],return_inverse=True)
    n=inverse.max()+1;rng=np.random.default_rng(3407);deltas=[]
    for _ in range(1000):
        weights=np.bincount(rng.integers(n,size=n),minlength=n)[inverse]
        if any(weights[y==label].sum()==0 for label in [0,1]):continue
        deltas.append(roc_auc_score(y,m,sample_weight=weights)-roc_auc_score(y,b,sample_weight=weights))
    return {'replicates':len(deltas),'unit':'paired video cluster','delta_ci95':np.quantile(deltas,[.025,.975]).tolist()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='pilot_v1');args=parser.parse_args()
    collected=[]
    for fold in ['A1_action_01','C1_composition_01']:
        dirs={arm:ROOT/'runs'/fold/arm/'seed3407'/args.run_id for arm in ['baseline','shine']}
        for arm,d in dirs.items():
            if json.loads((d/'status.json').read_text())['status']!='completed':
                raise RuntimeError('Incomplete arm '+str(d))
        results={arm:json.loads((d/'metrics.json').read_text()) for arm,d in dirs.items()}
        row={'fold':fold,'seed':3407,'baseline':results['baseline'],'shine':results['shine'],'comparison':{}}
        for role in ['seen','novel']:
            row['comparison'][role]={'delta_auroc':results['shine'][role]['pooled_auroc']-results['baseline'][role]['pooled_auroc'],
                **interval(read(dirs['baseline']/('predictions_'+role+'.jsonl')),read(dirs['shine']/('predictions_'+role+'.jsonl')))}
        collected.append(row)
    macro={role:float(np.mean([r['comparison'][role]['delta_auroc'] for r in collected])) for role in ['seen','novel']}
    report={'status':'completed','run_id':args.run_id,'seed':3407,'multi_seed':False,'formal_U_accessed':False,
        'selection':'Seen-val pooled AUROC; earliest tie','folds':collected,'macro_delta':macro,
        'limitations':['single seed / two exploratory inner folds','nominal existing GMR clip grid',
                      'additional supervision and forward budget differ','no temporal-vs-existence ablation yet']}
    (ROOT/'records/RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# Idea 10：单 seed 两折训练结果','',
           '已完成。Seed 3407，每 arm 10 epochs；按 Seen-val existence AUROC 选 epoch。Novel 为 strict-inner Novel-dev，正式 U 未参与。', '',
           '| Fold | Arm | Epoch | Seen AUROC | Novel AUROC | Raw R1@.5 Seen / Novel |',
           '|---|---|---:|---:|---:|---:|']
    for r in collected:
        for arm in ['baseline','shine']:
            m=r[arm];lines.append(f"| {r['fold']} | {arm} | {m['best_epoch']} | {m['seen']['pooled_auroc']:.4f} | {m['novel']['pooled_auroc']:.4f} | {m['seen']['raw_r1_iou_05']:.4f} / {m['novel']['raw_r1_iou_05']:.4f} |")
    lines+=['',f"两折等权 macro：SHINE−微调 baseline 的 Seen Δ={macro['seen']*100:+.2f} pp，Novel-dev Δ={macro['novel']*100:+.2f} pp。", '',
            '| Fold | Novel ΔAUROC (pp) | Paired video-cluster 95% CI (pp) |','|---|---:|---:|']
    for r in collected:
        c=r['comparison']['novel'];lo,hi=c['delta_ci95'];lines.append(f"| {r['fold']} | {c['delta_auroc']*100:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] |")
    lines+=['','上述 CI 是固定单 seed 训练结果的共享视频重采样，不包含重训不确定性；两折也不能视为独立域。', '',
        '损失、支持覆盖、条件排序、阈值、gated R@.5、显存及时间详见 [RESULTS.json](RESULTS.json)。Raw R@IoU 为直接 top1 筛查，不是 official mAP。', '',
        '本轮采用用户确认的 batch 轮换 absence 和强制编辑距离链；沿用当前 GMR 名义时间映射。额外监督/forward 预算与 baseline 不同，尚未拆分 temporal 与 existence loss 的机制增量。单 seed 两折不支持跨 backbone 稳定泛化结论。']
    (ROOT/'records/PILOT_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'macro_delta':macro,'folds':[{ 'fold':r['fold'],'comparison':r['comparison']} for r in collected]},indent=2))


if __name__=='__main__':main()
