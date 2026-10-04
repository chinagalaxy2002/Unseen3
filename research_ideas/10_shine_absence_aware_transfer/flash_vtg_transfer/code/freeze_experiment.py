"""Snapshot code, canonical sources, train/Seen rows and feature identities before training."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from train_flash_saliency import ARM_WEIGHTS,canonical_path
BASE=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval')
RELEASE=Path('/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def main():
    files={};features=set();splits={};provenance={}
    def add(p):files[str(p.resolve())]=sha(p)
    for p in (ROOT/'code').rglob('*.py'):add(p)
    for top in ['models','training']:
        for p in (ROOT/'code'/top/'flash_vtg_gmr').rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            source=BASE/top/'flash_vtg_gmr'/p.relative_to(ROOT/'code'/top/'flash_vtg_gmr')
            if sha(p)!=sha(source):raise RuntimeError('FlashVTG snapshot mismatch '+str(p))
            provenance[str(source)]=sha(source);add(p)
    add(ROOT/'configs/EXPERIMENT_PLAN.md');add(ROOT/'SOURCE_LICENSE');add(ROOT/'records/SMOKE_CHECK.json')
    parent_freeze=json.loads((ROOT.parent/'configs/FORMAL_FREEZE.json').read_text())
    for split in ['A1','C1']:
        dest=ROOT/'data/formal'/split
        train=rows(dest/'train.jsonl');val=rows(dest/'seen_val.jsonl')
        if {r['vid'] for r in train}&{r['vid'] for r in val}:raise RuntimeError('Video overlap')
        for r in train+val:
            if bool(r['exist_label'])!=bool(r.get('relevant_windows',[])):raise RuntimeError('Window/existence label mismatch')
        for name,key in [('train.jsonl','train_sha256'),('seen_val.jsonl','seen_val_sha256'),('query_bank.json','query_bank_sha256'),('split_spec_provenance.json','spec_sha256')]:
            p=dest/name
            if sha(p)!=parent_freeze['data'][split][key]:raise RuntimeError('Prepared data changed '+str(p))
            add(p)
        shutil.copyfile(RELEASE/split/'test.jsonl',dest/'test.jsonl');add(dest/'test.jsonl')
        source=canonical_path(split);add(source)
        ck=torch.load(source,map_location='cpu',weights_only=False);opt=ck['opt']
        for r in train+val:
            basepath=Path(opt.t_feat_dir)/f"qid{r['qid']}"
            choices=[basepath.with_suffix('.npz'),basepath.with_suffix('.pt')]
            p=next((x for x in choices if x.exists()),None)
            if p is None:raise RuntimeError('Missing text '+str(basepath))
            features.add(p)
            stem=Path(r['vid']).stem
            if stem.endswith('.mp4'):stem=stem[:-4]
            for folder in opt.v_feat_dirs:
                choices=[Path(folder)/(stem+'.npz'),Path(folder)/(stem+'.pt')]
                p=next((x for x in choices if x.exists()),None)
                if p is None:raise RuntimeError('Missing video '+stem)
                features.add(p)
        bank=json.loads((dest/'query_bank.json').read_text())
        for entry in bank.values():
            for edit in entry['edits']:features.add(ROOT.parent/'artifacts/edit_clip'/(edit['feature_id']+'.npz'))
        splits[split]={'train_rows':len(train),'seen_val_rows':len(val),'canonical_path':str(source),'canonical_sha256':sha(source),'checkpoint_original_epoch':ck.get('epoch'),'video_feature_paths':opt.v_feat_dirs,'text_feature_path':opt.t_feat_dir,'clip_length':opt.clip_length,'source_opt_wd':opt.wd,'source_opt_grad_clip':opt.grad_clip}
    print('Hashing read-only natural and edit feature files:',len(features),flush=True)
    for i,p in enumerate(sorted(features)):
        add(p)
        if (i+1)%10000==0:print('Hashed',i+1,flush=True)
    freeze={'created_unix':time.time(),'backbone':'FlashVTG-GMR','seed':3407,'epochs':50,'lr':1e-5,'batch_size':16,'arms':ARM_WEIGHTS,'selection':'earliest best trained Seen-val AUROC; no epoch0 fallback','formal_U_labels_read_for_design':False,'file_sha256':files,'source_code_provenance_sha256':provenance,'splits':splits,'feature_file_count':len(features),'model_snapshot':'physical copy; unmodified FlashVTG architecture; original use_neg retained','added_existence_supervision':False,'schedule':{'cuda:0':'A1 B0 and S1','cuda:1':'C1 B0 and S1'},'test_scope':'exploratory; post-completion frozen Seen-val selection; not used for hyperparameter choice'}
    with (ROOT/'configs/EXPERIMENT_FREEZE.json').open('x') as f:json.dump(freeze,f,indent=2)
    print('FlashVTG experiment frozen:',len(files),'files',flush=True)
if __name__=='__main__':main()
