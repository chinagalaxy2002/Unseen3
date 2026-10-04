"""Adopt running S1; launch E1/SE1 concurrently; preserve completed and live runs."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=['B0','S1_saliency_only','E1_rotated_exist_only','SE1_full']

def write(path,value):
    tmp=path.with_suffix('.parallel.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def state(split,arm):
    p=ROOT/f'runs/ablation_v2/{split}/{arm}/seed3407/component_v2_10ep/status.json'
    if not p.exists():return {'status':'queued'}
    try:return json.loads(p.read_text())
    except json.JSONDecodeError:return {'status':'writing'}

def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);p.add_argument('--device',required=True)
    a=p.parse_args();out=ROOT/'records/ablation_v2';started=time.time();children={};logs=[]
    freeze=json.loads((ROOT/'configs/ABLATION_V2_PARALLEL_FREEZE.json').read_text())
    for name,h in freeze['file_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise RuntimeError(f'Parallel freeze mismatch {name}')
    prior=freeze['adopted_workers'][a.fold];old=prior['trainer_pid']
    status=out/f'worker_{a.fold}.json'
    try:
        if state(a.fold,'S1_saliency_only')['status'] not in ('running','completed'):raise RuntimeError('Cannot adopt S1')
        for arm in ARMS[2:]:
            if state(a.fold,arm)['status']!='queued':raise RuntimeError(f'Arm already started: {arm}')
            log=(out/f'{a.fold}_{arm}_training_10ep.log').open('x');logs.append(log)
            children[arm]=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/train_ablation_v2_parallel.py'),
                    '--fold',a.fold,'--device',a.device,'--arms',arm],stdout=log,stderr=subprocess.STDOUT)
        def report(status_value='running',**extra):
            write(status,{'status':status_value,'fold':a.fold,'device':a.device,'pid':os.getpid(),
                  'adopted_s1_trainer_pid':old,'parallel_trainer_pids':{arm:proc.pid for arm,proc in children.items()},
                  'started':started,'runs':{arm:state(a.fold,arm) for arm in ARMS},**extra})
        report();retired=False;resumed=False
        while True:
            if not resumed and all(state(a.fold,arm)['status'] in ('running','completed') for arm in ARMS[2:]):
                os.kill(old,signal.SIGCONT);resumed=True
            s1=state(a.fold,'S1_saliency_only')
            if s1['status']=='completed' and not retired:
                # All S1 outputs and selected checkpoint are already committed. End old sequential launcher.
                if alive(old):
                    try:os.kill(old,signal.SIGTERM)
                    except ProcessLookupError:pass
                retired=True
            if s1['status']=='failed' or (not alive(old) and s1['status'] not in ('completed','writing')):
                raise RuntimeError('Adopted S1 failed before completion')
            for arm,proc in children.items():
                rc=proc.poll()
                if rc is not None and rc!=0:raise RuntimeError(f'{arm} failed: {rc}')
            report()
            if all(state(a.fold,arm)['status']=='completed' for arm in ARMS) and all(proc.poll()==0 for proc in children.values()):break
            time.sleep(2)
        subprocess.run([sys.executable,str(ROOT/'code/summarize_ablation_v2.py'),'--fold',a.fold],check=True)
        report('completed',elapsed_seconds=time.time()-started)
    except BaseException as exc:
        write(status,{'status':'failed','fold':a.fold,'error':repr(exc),'pid':os.getpid(),
              'parallel_trainer_pids':{arm:proc.pid for arm,proc in children.items()},'runs':{arm:state(a.fold,arm) for arm in ARMS}})
        raise
    finally:
        for log in logs:log.close()
