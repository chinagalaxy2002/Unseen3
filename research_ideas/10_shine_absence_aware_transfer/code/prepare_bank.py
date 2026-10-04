"""Match upstream SHINE edits to inner-train rows without consulting official U."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    import spacy
    from semantic_parser import canonicalize, predicate_components, SKIP_VERBS
    nlp = spacy.load('en_core_web_sm')
    upstream = {}
    for path in sorted((ROOT/'vendor/shine/data').glob('charades*/*gpt_train.jsonl')):
        for row in read(path):
            upstream[(row['vid'], ' '.join(row['query'].lower().split()))] = row
    unique, reports = {}, {}
    for fold in ['A1_action_01', 'C1_composition_01']:
        source = PROJECT/'experiments/moment_detr_gmr_evidence_v5/inner_folds'/fold
        local = ROOT/'data'/fold
        local.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((source/'manifest.json').read_text())
        inputs = {}
        for role in ['inner_train', 'inner_seen_val', 'inner_novel_dev']:
            entry = manifest['files'][role]
            if sha(entry['path']) != entry['sha256']:
                raise RuntimeError('Fold input hash mismatch')
            shutil.copyfile(entry['path'], local/(role+'.jsonl'))
            inputs[role] = {'source':entry['path'], 'sha256':entry['sha256']}
        shutil.copyfile(source/'manifest.json', local/'manifest.json')
        rows = read(local/'inner_train.jsonl')
        spec_path = Path('/home/guoxiangyu/paper/Openword/data/release/semantic_existence_v2')/manifest['parent_split']/'split_spec_provenance.json'
        spec = json.loads(spec_path.read_text())['split_spec']
        held_actions = set(spec['held_actions'])
        held_pairs = {tuple(p.split('|')) for p in spec['held_compositions']}
        if manifest['level']=='action':
            held_actions.update(manifest['heldout'])
        else:
            held_pairs.update(tuple(p.split('::')) for p in manifest['heldout'])
        # Metadata only: no official test rows/features are read.
        shutil.copyfile(spec_path, local/'outer_split_spec_provenance.json')
        candidates = []
        missing = 0
        for row in rows:
            if row['exist_label'] != 1:
                continue
            up = upstream.get((row['vid'], ' '.join(row['query'].lower().split())))
            if up is None or len(up.get('recomposed_queries', [])) != 3:
                missing += 1
                continue
            candidates.append((row, up))
        texts = sorted({text for _, up in candidates for text in up['recomposed_queries']})
        parsed = {}
        for text, doc in zip(texts, nlp.pipe(texts, batch_size=128)):
            graph = canonicalize(doc, {})
            verbs = {predicate_components(t)[1] for t in doc if t.pos_=='VERB' and t.lemma_.lower() not in SKIP_VERBS}
            nouns = {t.lemma_.lower() for t in doc if t.pos_ in {'NOUN','PROPN'}}
            if 'sofa' in nouns: nouns.add('couch')
            valid = not (verbs & held_actions) and not any((v,n) in held_pairs for v in verbs for n in nouns)
            parsed[text] = (valid, graph)
        bank, excluded = {}, 0
        for row, up in candidates:
            edits = up['recomposed_queries']
            if not all(parsed[t][0] for t in edits):
                excluded += 1
                continue
            entries = []
            for level, text in enumerate(edits):
                key = hashlib.sha256(text.encode()).hexdigest()[:24]
                unique[key] = text
                entries.append({'feature_id':key,'query':text,'level':level+1,'semantic_graph':parsed[text][1]})
            bank[str(row['qid'])] = {'source_qid':row['qid'],'upstream_qid':up['qid'],'vid':row['vid'], 'edits':entries}
        (local/'query_bank.json').write_text(json.dumps(bank, ensure_ascii=False, indent=2))
        reports[fold] = {'inputs':inputs, 'train_rows':len(rows), 'positive_rows':sum(r['exist_label'] for r in rows),
                         'eligible_three_level_chains':len(bank), 'missing_upstream':missing,
                         'excluded_semantics':excluded, 'semantic_filter':'all parsed content verbs against outer+inner held actions; conservative verb x noun check against held compositions',
                         'held_actions':sorted(held_actions),'held_pairs':sorted(held_pairs),'split_spec_sha256':sha(spec_path),
                         'bank_sha256':sha(local/'query_bank.json')}
    (ROOT/'data/unique_edit_texts.json').write_text(json.dumps(unique, ensure_ascii=False, indent=2))
    (ROOT/'records/QUERY_BANK_AUDIT.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    print(json.dumps(reports, indent=2), flush=True)


def encode(device):
    import sys
    sys.path.insert(0,str(ROOT/'code/_deps'))
    import numpy as np
    import torch
    import clip
    weights = Path('/home/guoxiangyu/Beyond_Caption-Based_Queries_for_Video_Moment_Retrieval/experiments_and_data/evaluations/flash_vtg_subset_consistency_eval/models/ViT-B-32.pt')
    model, _ = clip.load(str(weights), device=device, jit=False, text_only=True)
    model.eval()
    texts = json.loads((ROOT/'data/unique_edit_texts.json').read_text())
    out = ROOT/'artifacts/edit_clip'
    out.mkdir(parents=True, exist_ok=True)
    todo = [(key, text) for key, text in texts.items() if not (out/(key+'.npz')).exists()]
    with torch.no_grad():
        for start in range(0,len(todo),128):
            batch = todo[start:start+128]
            tokens = clip.tokenize([text for _,text in batch], context_length=77).to(device)
            states = model.encode_text(tokens)['last_hidden_state'].float().cpu().numpy()
            lengths = (tokens != 0).sum(1).cpu().tolist()
            for (key,text), state, length in zip(batch, states, lengths):
                np.savez_compressed(out/(key+'.npz'), last_hidden_state=state[:length])
            print(f'encoded {min(start+128,len(todo))}/{len(todo)}', flush=True)
    identities = {key:sha(out/(key+'.npz')) for key in texts}
    (ROOT/'records/TEXT_FEATURE_MANIFEST.json').write_text(json.dumps({'weights':str(weights),'weights_sha256':sha(weights),
       'code_sha256':sha(ROOT/'code/clip/model.py'),'features':identities,'normalization':'raw states, per-token L2 in loader'}, indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','encode'])
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    prepare() if args.stage=='prepare' else encode(args.device)
