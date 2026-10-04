"""Prepare only formal train and Seen validation; no formal test rows are read."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RELEASE=Path('/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2')
BASELINES=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/results/semantic_existence/multi_split_v2')
SPLITS=['A1','A2_alt','A3','C1','C2_alt']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in Path(p).open() if l.strip()]
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def prepare():
    import spacy
    from semantic_parser import canonicalize,predicate_components,SKIP_VERBS
    nlp=spacy.load('en_core_web_sm')
    upstream={}
    for path in sorted((ROOT/'vendor/shine/data').glob('charades*/*gpt_train.jsonl')):
        for row in read(path):upstream[(row['vid'],' '.join(row['query'].lower().split()))]=row
    parsed={};unique={};report={}
    for split in SPLITS:
        out=ROOT/'data/formal'/split;out.mkdir(parents=True,exist_ok=True)
        source=RELEASE/split/'train.jsonl'
        seen_source=Path('/home/guoxiangyu/paper/Openword/generalized-moment-retrieval/features/semantic_existence_v2')/split/'val_seen.jsonl'
        rows=read(source);seen=read(seen_source)
        if any(r['partition'] not in ['S+','S-'] for r in rows+seen):raise RuntimeError('Non-Seen training/selection input')
        shutil.copyfile(source,out/'train.jsonl');shutil.copyfile(seen_source,out/'seen_val.jsonl')
        spec_path=RELEASE/split/'split_spec_provenance.json'
        spec=json.loads(spec_path.read_text())['split_spec']
        shutil.copyfile(spec_path,out/'split_spec_provenance.json')
        held_actions=set(spec['held_actions']);held_pairs={tuple(p.split('|')) for p in spec['held_compositions']}
        candidates=[];missing=0
        for row in rows:
            if not row['exist_label']:continue
            u=upstream.get((row['vid'],' '.join(row['query'].lower().split())))
            if u is None or len(u.get('recomposed_queries',[]))!=3:missing+=1;continue
            candidates.append((row,u))
        todo=sorted({t for _,u in candidates for t in u['recomposed_queries']} - set(parsed))
        for text,doc in zip(todo,nlp.pipe(todo,batch_size=128)):
            verbs={predicate_components(t)[1] for t in doc if t.pos_=='VERB' and t.lemma_.lower() not in SKIP_VERBS}
            nouns={t.lemma_.lower() for t in doc if t.pos_ in ['NOUN','PROPN']}
            if 'sofa' in nouns:nouns.add('couch')
            parsed[text]={'graph':canonicalize(doc,{}),'verbs':verbs,'nouns':nouns}
        bank={};excluded=0
        for row,u in candidates:
            edits=u['recomposed_queries']
            if any(parsed[t]['verbs']&held_actions or any((v,n) in held_pairs for v in parsed[t]['verbs'] for n in parsed[t]['nouns']) for t in edits):
                excluded+=1;continue
            entries=[]
            for level,text in enumerate(edits):
                key=hashlib.sha256(text.encode()).hexdigest()[:24];unique[key]=text
                entries.append({'feature_id':key,'query':text,'level':level+1,'semantic_graph':parsed[text]['graph']})
            bank[str(row['qid'])]={'source_qid':row['qid'],'upstream_qid':u['qid'],'vid':row['vid'],'edits':entries}
        write(out/'query_bank.json',bank)
        report[split]={'train_rows':len(rows),'seen_val_rows':len(seen),'positive_train':sum(r['exist_label'] for r in rows),
            'eligible_chains':len(bank),'missing_upstream':missing,'excluded_held_semantics':excluded,
            'train_sha256':sha(source),'seen_val_sha256':sha(seen_source),'query_bank_sha256':sha(out/'query_bank.json'),
            'spec_sha256':sha(spec_path),'held_actions':sorted(held_actions),'held_pairs':sorted(held_pairs),
            'source_checkpoint':str(BASELINES/split/'moment/best.ckpt'),'checkpoint_sha256':sha(BASELINES/split/'moment/best.ckpt'),
            'formal_test_read':False}
        print(split,report[split],flush=True)
    write(ROOT/'data/formal/unique_edit_texts.json',unique)
    write(ROOT/'records/formal/DATA_MANIFEST.json',report)


def encode(device):
    import numpy as np
    import torch
    sys.path.insert(0,str(ROOT/'code/_deps'))
    import clip
    weights=Path('/home/guoxiangyu/Beyond_Caption-Based_Queries_for_Video_Moment_Retrieval/experiments_and_data/evaluations/flash_vtg_subset_consistency_eval/models/ViT-B-32.pt')
    model,_=clip.load(str(weights),device=device,jit=False,text_only=True);model.eval()
    texts=json.loads((ROOT/'data/formal/unique_edit_texts.json').read_text())
    out=ROOT/'artifacts/edit_clip';out.mkdir(parents=True,exist_ok=True)
    todo=[(key,t) for key,t in texts.items() if not (out/(key+'.npz')).exists()]
    with torch.no_grad():
        for start in range(0,len(todo),128):
            batch=todo[start:start+128];tokens=clip.tokenize([t for _,t in batch]).to(device)
            states=model.encode_text(tokens)['last_hidden_state'].float().cpu().numpy();lengths=(tokens!=0).sum(1).cpu().tolist()
            for (key,_),state,length in zip(batch,states,lengths):np.savez_compressed(out/(key+'.npz'),last_hidden_state=state[:length])
            print(f'encoded {min(start+128,len(todo))}/{len(todo)}',flush=True)
    write(ROOT/'records/formal/TEXT_FEATURE_MANIFEST.json',{'weights_sha256':sha(weights),'features':{key:sha(out/(key+'.npz')) for key in texts}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','encode']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    prepare() if a.stage=='prepare' else encode(a.device)
