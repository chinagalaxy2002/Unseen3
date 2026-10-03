#!/usr/bin/env python
"""Atomic split jobs; a below-floor fallback is saved and labelled for evaluation."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];gpu=int(sys.argv[1]);splits=sys.argv[2:]
for split in splits:
 out=ROOT/'results/moment_detr_trm_gmr_joint_v3'/split;out.mkdir(parents=True,exist_ok=True)
 existing=out/'joint_v3_summary.json'
 if existing.exists():
  state=json.loads(existing.read_text())['status']
  if state == 'completed':
   print(f'Skip already finished {split}: {state}',flush=True);continue
 print(f'Start {split} on GPU {gpu}',flush=True)
 with (out/'job.log').open('a') as log:
  status=subprocess.run(['bash','scripts/train_trm_gmr_joint_v3.sh',split,str(gpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
 record={'split':split,'gpu':gpu,'returncode':status,'status':'finished' if status==0 else 'failed','ended_unix_time':time.time()}
 (out/'job_status.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True)
