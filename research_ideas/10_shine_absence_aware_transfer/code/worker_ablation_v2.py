"""Detached, fail-fast A1/C1 worker; no formal test access."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);p.add_argument('--device',required=True)
    a=p.parse_args();out=ROOT/'records/ablation_v2';out.mkdir(parents=True,exist_ok=True)
    status=out/f'worker_{a.fold}.json';started=time.time()
    with status.open('x') as f:
        json.dump({'status':'running','fold':a.fold,'device':a.device,'pid':os.getpid(),'started':started},f,indent=2)
    try:
        with (out/f'{a.fold}_training_10ep.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/train_ablation_v2.py'),'--fold',a.fold,'--device',a.device],stdout=log,stderr=subprocess.STDOUT)
            write(status,{'status':'running','fold':a.fold,'device':a.device,'pid':os.getpid(),'trainer_pid':proc.pid,'started':started})
            rc=proc.wait()
        if rc: raise RuntimeError(f'Trainer exited {rc}')
        subprocess.run([sys.executable,str(ROOT/'code/summarize_ablation_v2.py'),'--fold',a.fold],check=True)
        write(status,{'status':'completed','fold':a.fold,'elapsed_seconds':time.time()-started})
    except BaseException as exc:
        write(status,{'status':'failed','fold':a.fold,'error':repr(exc),'elapsed_seconds':time.time()-started})
        raise
