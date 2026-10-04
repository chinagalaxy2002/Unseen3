"""Run fixed split/arm jobs sequentially on one GPU; stop on failure."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--splits',nargs='+',required=True)
    a=p.parse_args();out=ROOT/'records/formal';out.mkdir(parents=True,exist_ok=True)
    name=a.device.replace(':','_');status=out/('worker_'+name+'.json')
    started=time.time();done=[]
    for split in a.splits:
        log=out/(split+'_training_50ep.log')
        status.write_text(json.dumps({'status':'running','device':a.device,'current':split,'completed_splits':done,'started':started},indent=2))
        with log.open('w') as f:
            proc=subprocess.run([sys.executable,str(ROOT/'code/train_formal.py'),'--fold',split,'--device',a.device,
                    '--epochs','50','--run-id','formal_v1_50ep'],stdout=f,stderr=subprocess.STDOUT)
        if proc.returncode:
            status.write_text(json.dumps({'status':'failed','current':split,'returncode':proc.returncode,'completed_splits':done},indent=2))
            sys.exit(proc.returncode)
        done.append(split)
        print('COMPLETED SPLIT '+split,flush=True)
    status.write_text(json.dumps({'status':'completed','device':a.device,'completed_splits':done,'elapsed_seconds':time.time()-started},indent=2))
