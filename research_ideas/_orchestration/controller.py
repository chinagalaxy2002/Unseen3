#!/usr/bin/env python3
"""Bounded queue supervisor; starts workers and performs at most two scoped CLI repairs/task."""
from __future__ import annotations
import argparse, hashlib, json, os, shlex, sqlite3, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
DB=HERE/'queue.sqlite3'; STATE=HERE/'controller_state.json'; LOG=HERE/'controller.log'

def now():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def connect():
 c=sqlite3.connect(DB,timeout=30);c.row_factory=sqlite3.Row;return c
def proc_start(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
 except (OSError,IndexError,TypeError):return None
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def state_read():
 return json.loads(STATE.read_text()) if STATE.exists() else {'worker_restarts':{},'repairs':{},'started_at':now()}
def state_write(s):STATE.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
def log(msg):
 with LOG.open('a') as f:f.write(f'[{now()}] {msg}\n')
def task_rows():
 with connect() as c:return [dict(x) for x in c.execute('SELECT * FROM tasks ORDER BY created_at,id')]
def active_worker(gpu):
 with connect() as c:
  rows=[dict(x) for x in c.execute("SELECT * FROM workers WHERE resource=? AND state='running'",(f'gpu:{gpu}',))]
 return [x for x in rows if proc_start(x['pid'])==x['start_token']]
def start_worker(gpu,state,max_restarts):
 live=active_worker(gpu)
 if live:return True
 key=str(gpu);n=state['worker_restarts'].get(key,0)
 if n>=max_restarts:
  log(f'worker restart budget exhausted on GPU {gpu}');return False
 name=f'research_queue_gpu{gpu}_{int(time.time())}'
 repo=Path(__file__).resolve().parents[2]
 cmd=f"cd {shlex.quote(str(repo))} && {shlex.quote(sys.executable)} research_ideas/_orchestration/orchestrator.py worker --gpu {shlex.quote(str(gpu))} --idle-exit 30 >> research_ideas/_orchestration/worker_gpu{gpu}.log 2>&1"
 r=subprocess.run(['tmux','new-session','-d','-s',name,'-c',str(repo),cmd],capture_output=True,text=True)
 if r.returncode:
  log(f'worker start failed GPU={gpu}: {r.stderr.strip()}');return False
 state['worker_restarts'][key]=n+1;state_write(state);log(f'started GPU worker {gpu} session={name} start_count={n+1}');return True
def make_repair_spec(task,attempt,repair_record):
 oldpath=Path(json.loads(task['config_json'])['task_spec_file'])
 spec=json.loads(oldpath.read_text());oldrun=Path(spec['train_log']).parent
 old_config_path=Path(spec['config']);old_config=json.loads(old_config_path.read_text())
 if oldrun.name=='runs':raise RuntimeError('cannot resolve isolated run root')
 idea=Path(__file__).resolve().parents[1]/task['idea']
 new_id=f"{task['id']}_repair{attempt}";newrun=idea/'runs'/task['fold']/new_id
 newrun.mkdir(parents=True,exist_ok=False)
 def replace_run(value):
  if isinstance(value,str):return value.replace(str(oldrun),str(newrun))
  if isinstance(value,list):return [replace_run(x) for x in value]
  if isinstance(value,dict):return {k:replace_run(v) for k,v in value.items()}
  return value
 spec=replace_run(spec);spec['id']=new_id;spec['priority']=int(spec.get('priority',0))+1
 spec['config']=str(newrun/'resolved_config.json');spec['train_log']=str(newrun/'train.log');spec['eval_log']=str(newrun/'eval.log')
 spec['parent_task']=task['id'];spec['repair_attempt']=attempt;spec['repair_record']=str(repair_record)
 cfg=replace_run(old_config)
 cfg.update(task_id=new_id,parent_task=task['id'],repair_attempt=attempt,repair_record=str(repair_record))
 Path(spec['config']).write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
 specpath=HERE/'specs'/f'{new_id}.json';specpath.write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
 return specpath
def repair_task(task,state,max_repairs,timeout):
 key=task['id'];n=state['repairs'].get(key,0)
 if n>=max_repairs:return False
 cfg=json.loads(task['config_json']);specfile=cfg.get('task_spec_file')
 if not specfile or not Path(specfile).is_file():log(f'cannot repair {key}: missing task spec provenance');return False
 spec=json.loads(Path(specfile).read_text());idea=Path(__file__).resolve().parents[1]/task['idea']
 attempt=n+1;state['repairs'][key]=attempt;state_write(state)
 recdir=idea/'records'/'repairs';recdir.mkdir(parents=True,exist_ok=True)
 record=recdir/f'{key}_repair{attempt}.md';last=recdir/f'{key}_repair{attempt}_last_message.md';cli_log=recdir/f'{key}_repair{attempt}_cli.log'
 before=subprocess.run(['git','-C',str(Path(__file__).resolve().parents[2]),'status','--porcelain=v1','--untracked-files=all'],capture_output=True,text=True).stdout.splitlines()
 prompt=(f"A queued experiment failed with a software/validation error. Inspect only this failed task's log at {task['eval_log']} and the isolated experiment code under {idea/'code'}. "
  f"Make the smallest repair only in {idea/'code'}; do not edit source baselines, shared data, other ideas, protocol/freeze files, or existing result files. Do not run training or evaluation. "
  f"Add a concise repair record at {record} explaining the observed error and exact code change. The next attempt will use a new task ID/config/output directory. If the log indicates insufficient labels/data rather than an engineering error, make no change and state that.")
 cmd=['codex','exec','--cd',str(idea),'--sandbox','workspace-write','--output-last-message',str(last),prompt]
 try:
  with cli_log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,text=True,timeout=timeout)
 except subprocess.TimeoutExpired:
  log(f'Codex repair timed out task={key} after {timeout}s');return False
 if r.returncode!=0:
  log(f'Codex repair exited rc={r.returncode} task={key}; see {cli_log}');return False
 after=subprocess.run(['git','-C',str(Path(__file__).resolve().parents[2]),'status','--porcelain=v1','--untracked-files=all'],capture_output=True,text=True).stdout.splitlines()
 changed=set(after)-set(before)
 outside=[line for line in changed if f'research_ideas/{task["idea"]}/code/' not in line and f'research_ideas/{task["idea"]}/records/repairs/' not in line]
 if outside:
  log(f'repair rejected due changed paths outside authorized idea code/repair record: {outside}');return False
 # Syntax check changed Python sources and freeze the repair record before any task retry.
 pyfiles=list((idea/'code').glob('*.py'))
 comp=subprocess.run([sys.executable,'-m','py_compile',*[str(p) for p in pyfiles]],capture_output=True,text=True)
 if comp.returncode:
  log(f'repair syntax validation failed task={key}: {comp.stderr[-1000:]}');return False
 new_spec=make_repair_spec(task,attempt,record)
 log(f'Codex repair accepted task={key}; creating new isolated config from {new_spec}')
 subprocess.run([sys.executable,str(HERE/'orchestrator.py'),'enqueue',str(new_spec)],check=True)
 return True
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpus',default='0');p.add_argument('--poll-seconds',type=int,default=15)
 p.add_argument('--idle-grace-seconds',type=int,default=45);p.add_argument('--max-worker-restarts',type=int,default=2)
 p.add_argument('--max-repairs-per-task',type=int,default=2);p.add_argument('--codex-timeout-seconds',type=int,default=600)
 a=p.parse_args();gpus=[x.strip() for x in a.gpus.split(',') if x.strip()];state=state_read();empty_since=None
 while True:
  subprocess.run([sys.executable,str(HERE/'orchestrator.py'),'recover'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  tasks=task_rows();pending=[t for t in tasks if t['status'] in {'ready','running','trained','evaluating'}]
  # One bounded CLI diagnosis/fix attempt follows deterministic retries and artifact validation.
  failed=[t for t in tasks if t['status']=='failed' and t['error_kind']=='command_or_validation']
  repaired_any=False
  for t in failed:
   if state['repairs'].get(t['id'],0)<a.max_repairs_per_task:
    try:repaired_any=repair_task(t,state,a.max_repairs_per_task,a.codex_timeout_seconds) or repaired_any
    except Exception as e:log(f'repair handler error task={t["id"]}: {e!r}')
  tasks=task_rows();pending=[t for t in tasks if t['status'] in {'ready','running','trained','evaluating'}]
  repairable=[t for t in tasks if t['status']=='failed' and t['error_kind']=='command_or_validation' and state['repairs'].get(t['id'],0)<a.max_repairs_per_task]
  if not pending and not repairable:
   if empty_since is None:empty_since=time.monotonic()
   if time.monotonic()-empty_since>=a.idle_grace_seconds:log('queue empty; bounded controller exiting');break
  else:empty_since=None
  for gpu in gpus:
   relevant=[t for t in pending if str(t.get('gpu'))==gpu]
   if relevant and not active_worker(gpu):start_worker(gpu,state,a.max_worker_restarts)
  state_write(state);time.sleep(a.poll_seconds)
if __name__=='__main__':main()
