#!/usr/bin/env python
"""Freeze all five splits using only released Seen train/val and Seen reference."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.moment_detr_trm_gmr_joint_v3.semantic_groups import SemanticGroupBatchSampler,eligible_validation_groups
SPLITS=['A1','A2_alt','A3','C1','C2_alt']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 out=ROOT/'experiments/trm_gmr_joint_v3';out.mkdir(parents=True,exist_ok=True)
 for split in SPLITS:
  release=ROOT/'data/release/semantic_existence_v2'/split
  def seen(name):
   rows=[json.loads(l) for l in (release/name).read_text().splitlines()]
   return [r for r in rows if r['partition'] in ('S+','S-')]
  tr,va=seen('train.jsonl'),seen('val.jsonl')
  for row in tr+va:
   for path in [ROOT/'features/semantic_existence_v2/shared_clip_text'/f"qid{row['qid']}.npz",ROOT/'features/phrase_data/clip_phrase'/f"qid{row['qid']}.npz",ROOT/'features/charades_video/vid_slowfast'/f"{row['vid']}.npz",ROOT/'features/charades_video/vid_clip'/f"{row['vid']}.npz"]:
    if not path.is_file():raise FileNotFoundError(path)
  ref_path=ROOT/'results/moment_detr_trm_gmr_joint'/split/'best_charades_sta_semantic_novelty_val_preds_metrics.json'
  ref=json.loads(ref_path.read_text())['brief']['MR-full-mAP']
  sampler=SemanticGroupBatchSampler(tr)
  eligible=eligible_validation_groups(tr,va,5)
  manifest={'split':split,'semantic_source':'frozen released semantic_graph; action_base and object_concept/object','sampler_audit':sampler.audit(),
   'validation_selection_groups':eligible,'validation_min_per_class':5,'localization_reference_mAP':ref,'localization_floor_mAP':ref-1.0,
   'reference_source':str(ref_path.relative_to(ROOT)),'reference_sha256':digest(ref_path),'train_sha256':digest(release/'train.jsonl'),'val_sha256':digest(release/'val.jsonl'),
   'train_queries':len(tr),'val_queries':len(va),'missing_feature_or_fallback':False,'U_accessed':False}
  (out/f'{split}_semantic_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
  print(split, 'Seen reference',ref,'floor',ref-1,'eligible selection action/comp',len(eligible['action']),len(eligible['composition']))
if __name__=='__main__':main()
