"""Two matched 50-epoch arms per GPU; automatically evaluate after completion."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=['B0','S1_saliency_only']

def state(split,arm):
    p=ROOT/f'runs/saliency_v5/{split}/{arm}/seed3407/saliency_v5_50ep/status.json'
    if not p.exists():return {'status':'queued'}
    try:return json.loads(p.read_text())
    except json.JSONDecodeError:return {'status':'writing'}

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);p.add_argument('--device',required=True);a=p.parse_args()
    out=ROOT/'records/saliency_v5';out.mkdir(parents=True,exist_ok=True);status=out/f'worker_{a.fold}.json';started=time.time();children={};logs=[]
    with status.open('x') as f:json.dump({'status':'starting','pid':os.getpid(),'fold':a.fold},f)
    def report(value='running',**extra):
        write(status,{'status':value,'fold':a.fold,'device':a.device,'pid':os.getpid(),'started':started,
             'trainer_pids':{arm:proc.pid for arm,proc in children.items()},'runs':{arm:state(a.fold,arm) for arm in ARMS},**extra})
    try:
        freeze=json.loads((ROOT/'configs/SALIENCY_V5_FREEZE.json').read_text())
        for name,h in freeze['file_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise RuntimeError(f'Freeze mismatch {name}')
        if any(state(a.fold,arm)['status']!='queued' for arm in ARMS):raise RuntimeError('Existing saliency v5 run; refuse duplicate')
        for arm in ARMS:
            log=(out/f'{a.fold}_{arm}_training_50ep.log').open('x');logs.append(log)
            children[arm]=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/train_saliency_v5.py'),'--fold',a.fold,'--device',a.device,'--arms',arm],stdout=log,stderr=subprocess.STDOUT)
        report()
        while True:
            codes={arm:proc.poll() for arm,proc in children.items()}
            if all(rc is not None for rc in codes.values()):
                if any(rc!=0 for rc in codes.values()):raise RuntimeError(f'Trainer exit codes: {codes}')
                break
            report();time.sleep(2)
        report('evaluating')
        subprocess.run([sys.executable,'-u',str(ROOT/'code/evaluate_saliency_v5.py'),'--fold',a.fold,'--device',a.device],check=True)
        report('completed',elapsed_seconds=time.time()-started)
        import fcntl
        with (out/'aggregate.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if all((out/f'evaluation/{split}/status.json').exists() and json.loads((out/f'evaluation/{split}/status.json').read_text())['status']=='completed' for split in ['A1','C1']):
                subprocess.run([sys.executable,str(ROOT/'code/summarize_saliency_v5.py')],check=True)
    except BaseException as exc:
        report('failed',error=repr(exc));raise
    finally:
        for log in logs:log.close()
