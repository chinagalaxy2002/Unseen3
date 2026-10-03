#!/usr/bin/env python
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((ROOT/'experiments/trm_gmr_joint_v3/EXPERIMENT_FREEZE.json').read_text())
if not f.get('code_git_sha'):raise ValueError('Code SHA must be frozen before training')
for path,expected in f['source_sha256'].items():
 if sha(ROOT/path)!=expected:raise ValueError(f'Frozen source changed: {path}')
split=sys.argv[1];m=json.loads((ROOT/f'experiments/trm_gmr_joint_v3/{split}_semantic_manifest.json').read_text())
for name in ['train','val']:
 if sha(ROOT/f'data/release/semantic_existence_v2/{split}/{name}.jsonl')!=m[f'{name}_sha256']:raise ValueError(f'Split data changed: {split}/{name}')
if sha(ROOT/m['reference_source'])!=m['reference_sha256']:raise ValueError('Seen reference changed')
print(f'Joint-v3 frozen source and Seen-only manifest verified for {split}',flush=True)
