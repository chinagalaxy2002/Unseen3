"""Wait for a quiet GPU before launching the independent FlashVTG experiment."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def gpu_processes(index):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    mapping={row.split(',')[0].strip():row.split(',')[1].strip() for row in rows}
    uuid=mapping[str(index)]
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    return [int(row.split(',')[1].strip()) for row in rows if row.split(',')[0].strip()==uuid]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',choices=['A1','C1'],required=True);p.add_argument('--device',choices=['cuda:0','cuda:1'],required=True);a=p.parse_args()
    status=ROOT/'records'/f'queue_{a.fold}.json';started=time.time();quiet_since=None;child=None
    with status.open('x') as f:json.dump({'status':'starting','pid':os.getpid(),'fold':a.fold,'device':a.device},f)
    try:
        freeze=json.loads((ROOT/'configs/EXPERIMENT_FREEZE.json').read_text())
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=freeze['file_sha256'][str(Path(__file__).resolve())]:raise RuntimeError('Queue source freeze mismatch')
        while True:
            occupying=gpu_processes(a.device.split(':')[1]);now=time.time()
            if occupying:quiet_since=None
            elif quiet_since is None:quiet_since=now
            write(status,{'status':'waiting_for_gpu','fold':a.fold,'device':a.device,'pid':os.getpid(),'started':started,'occupying_compute_pids':occupying,'quiet_since':quiet_since,'quiet_seconds_required':60,'planned_arms':['B0','S1_saliency_only'],'epochs':50})
            if quiet_since is not None and now-quiet_since>=60:break
            time.sleep(15)
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/worker_flash.py'),'--fold',a.fold,'--device',a.device])
        write(status,{'status':'running','fold':a.fold,'device':a.device,'pid':os.getpid(),'worker_pid':child.pid,'started':started,'wait_seconds':time.time()-started})
        code=child.wait()
        if code:raise RuntimeError(f'FlashVTG worker exited {code}')
        write(status,{'status':'completed','fold':a.fold,'device':a.device,'elapsed_seconds':time.time()-started})
    except BaseException as exc:
        write(status,{'status':'failed','fold':a.fold,'device':a.device,'error':repr(exc),'pid':os.getpid(),'worker_pid':child.pid if child else None})
        raise
