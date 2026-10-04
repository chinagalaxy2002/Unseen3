#!/usr/bin/env python3
"""Durable, single-owner-per-GPU experiment queue for this research workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "queue.sqlite3"
RUNS = ROOT / "worker_runs"
STATUSES = {"planned", "ready", "running", "trained", "evaluating", "succeeded", "failed", "blocked", "insufficient_evidence"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def proc_start(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[21]
    except (OSError, IndexError):
        return None


def proc_cmd(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def task_child_is_live(task):
    """Only defer recovery for the recorded child when PID, start token, task env and argv agree."""
    pid=task.get("child_pid"); token=task.get("child_start")
    if not pid or not token or proc_start(pid)!=token:
        return False
    try:
        raw=Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        env={x.split(b"=",1)[0].decode():x.split(b"=",1)[1].decode(errors="replace") for x in raw if b"=" in x}
        if env.get("RESEARCH_TASK_ID")!=task["id"]:
            return False
        actual=shlex.split(proc_cmd(pid))
        cfg=json.loads(task["config_json"])
        return any(actual[:len(expected)]==[str(x) for x in expected]
                   for expected in (cfg.get("train_cmd",[]),cfg.get("eval_cmd",[])) if expected)
    except (OSError,ValueError,KeyError,TypeError):
        return False


def dbconn():
    c = sqlite3.connect(DB, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=FULL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def init_db():
    ROOT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    with dbconn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, priority INTEGER NOT NULL, status TEXT NOT NULL,
          idea TEXT NOT NULL, backbone TEXT NOT NULL, fold TEXT NOT NULL,
          seed INTEGER NOT NULL, config_json TEXT NOT NULL, config_hash TEXT NOT NULL,
          code_hash TEXT NOT NULL, data_version TEXT NOT NULL, gpu TEXT,
          worker_id TEXT, worker_pid INTEGER, worker_start TEXT, child_pid INTEGER,
          child_start TEXT, attempt INTEGER NOT NULL DEFAULT 0, max_retries INTEGER NOT NULL DEFAULT 2,
          created_at TEXT NOT NULL, started_at TEXT, trained_at TEXT, ended_at TEXT,
          heartbeat_at TEXT, train_log TEXT, eval_log TEXT, checkpoint TEXT,
          predictions TEXT, metrics TEXT, error TEXT, error_kind TEXT,
          FOREIGN KEY(worker_id) REFERENCES workers(id)
        );
        CREATE TABLE IF NOT EXISTS workers (
          id TEXT PRIMARY KEY, resource TEXT NOT NULL, pid INTEGER NOT NULL,
          start_token TEXT NOT NULL, host TEXT NOT NULL, started_at TEXT NOT NULL,
          heartbeat_at TEXT NOT NULL, task_id TEXT, state TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_worker_per_resource ON workers(resource) WHERE state='running';
        CREATE INDEX IF NOT EXISTS task_ready_order ON tasks(status, priority DESC, created_at);
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
          at TEXT NOT NULL, old_status TEXT, new_status TEXT, detail_json TEXT NOT NULL
        );
        """)


def record_event(c, task_id, old, new, detail):
    c.execute("INSERT INTO events(task_id,at,old_status,new_status,detail_json) VALUES(?,?,?,?,?)",
              (task_id, now(), old, new, json.dumps(detail, sort_keys=True)))


def transition(c, task_id, expected, status, **fields):
    if status not in STATUSES:
        raise ValueError(status)
    row = c.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None or row["status"] not in expected:
        raise RuntimeError(f"Invalid transition {task_id}: {None if row is None else row['status']} -> {status}")
    old = row["status"]
    sets = ["status=?"]
    values = [status]
    for k, v in fields.items():
        sets.append(f"{k}=?")
        values.append(v)
    values.append(task_id)
    c.execute(f"UPDATE tasks SET {','.join(sets)} WHERE id=?", values)
    record_event(c, task_id, old, status, fields)


def enqueue(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    required = ["id", "idea", "backbone", "fold", "seed", "config", "code_files", "data_version", "gpu", "train_cmd", "eval_cmd", "checkpoint", "predictions", "metrics"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise ValueError(f"Missing fields in job spec: {missing}")
    config_path = Path(spec["config"]).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    ch = hashlib.sha256()
    code_hashes = {}
    for p0 in spec["code_files"]:
        p = Path(p0).resolve()
        code_hashes[str(p)] = digest(p)
        ch.update(str(p).encode()); ch.update(code_hashes[str(p)].encode())
    config = json.loads(config_path.read_text())
    config["task_spec_file"] = str(Path(spec_path).resolve())
    config["train_cmd"] = spec["train_cmd"]
    config["eval_cmd"] = spec["eval_cmd"]
    config["source_file"] = str(config_path)
    config["source_sha256"] = digest(config_path)
    config["code_file_hashes"] = code_hashes
    encoded = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(encoded.encode()).hexdigest()
    task_id = spec["id"]
    init_db()
    with dbconn() as c:
        if c.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
            raise RuntimeError(f"Task already exists: {task_id}")
        c.execute("""INSERT INTO tasks(id,priority,status,idea,backbone,fold,seed,config_json,config_hash,code_hash,
          data_version,gpu,created_at,max_retries,train_log,eval_log,checkpoint,predictions,metrics)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (task_id, int(spec.get("priority", 0)), "planned", spec["idea"], spec["backbone"], spec["fold"],
           int(spec["seed"]), encoded, config_hash, ch.hexdigest(), spec["data_version"], str(spec["gpu"]), now(),
           min(2, max(0, int(spec.get("max_retries", 2)))), spec["train_log"], spec["eval_log"], spec["checkpoint"],
           spec["predictions"], spec["metrics"]))
        record_event(c, task_id, None, "planned", {"spec": str(Path(spec_path).resolve()), "code_hash": ch.hexdigest()})
        transition(c, task_id, {"planned"}, "ready")
    print(task_id)


def gpu_is_idle(gpu):
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                           text=True, capture_output=True, check=True, timeout=10)
        rows = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        # Graphics/UI processes do not appear in compute-app query. Any compute process blocks a new task.
        return len(rows) == 0, rows
    except Exception as e:
        return False, [f"nvidia-smi query failed: {e!r}"]


def claim(gpu, worker_id, pid, start_token):
    with dbconn() as c:
        c.execute("BEGIN IMMEDIATE")
        existing = c.execute("SELECT id FROM workers WHERE resource=? AND state='running' AND id<>?", (f"gpu:{gpu}",worker_id)).fetchone()
        mine = c.execute("SELECT id FROM workers WHERE id=? AND state='running'",(worker_id,)).fetchone()
        if existing or not mine:
            c.rollback(); return None
        row = c.execute("SELECT * FROM tasks WHERE status='ready' AND gpu=? ORDER BY priority DESC,created_at,id LIMIT 1", (str(gpu),)).fetchone()
        if row is None:
            c.execute("UPDATE workers SET heartbeat_at=?,task_id=NULL WHERE id=?", (now(),worker_id)); c.commit(); return None
        ok, active = gpu_is_idle(gpu)
        if not ok:
            c.execute("UPDATE workers SET state='idle' WHERE id=?", (worker_id,)); c.commit()
            print(f"GPU {gpu} no longer idle; task remains ready: {active}", flush=True)
            return "busy"
        c.execute("UPDATE workers SET task_id=?,heartbeat_at=? WHERE id=?", (row["id"], now(), worker_id))
        c.execute("UPDATE tasks SET status='running',worker_id=?,worker_pid=?,worker_start=?,started_at=COALESCE(started_at,?),heartbeat_at=?,attempt=attempt+1,gpu=? WHERE id=? AND status='ready'",
                  (worker_id,pid,start_token,now(),now(),str(gpu),row["id"]))
        record_event(c, row["id"], "ready", "running", {"worker":worker_id,"pid":pid,"gpu":str(gpu),"attempt":row["attempt"]+1})
        result = c.execute("SELECT * FROM tasks WHERE id=?", (row["id"],)).fetchone()
        c.commit()
        return dict(result)


def heartbeat(worker_id, task_id, child=None):
    with dbconn() as c:
        c.execute("UPDATE workers SET heartbeat_at=? WHERE id=? AND state='running'", (now(),worker_id))
        if task_id:
            c.execute("UPDATE tasks SET heartbeat_at=?,child_pid=?,child_start=? WHERE id=? AND status IN ('running','evaluating')",
                      (now(), child[0] if child else None, child[1] if child else None, task_id))


def validate_checkpoint(task):
    p=Path(task["checkpoint"])
    if not p.is_file() or p.stat().st_size <= 1024: return False
    try:
        import torch
        state=torch.load(p,map_location="cpu",weights_only=False)
        return isinstance(state,dict) and "model" in state and "best_epoch" in state and "best_seen_val_auroc" in state
    except Exception:
        return False


def validate_eval(task):
    cp, pr, mt = Path(task["checkpoint"]), Path(task["predictions"]), Path(task["metrics"])
    if not cp.is_file() or cp.stat().st_size <= 1024 or not pr.is_file() or pr.stat().st_size <= 20 or not mt.is_file():
        return False, "missing or empty expected artifacts"
    try:
        m=json.loads(mt.read_text())
        p=[json.loads(x) for x in pr.read_text().splitlines() if x]
        if not p or m.get("prediction_rows") != len(p): return False, "prediction coverage count mismatch"
        if not m.get("novel_dev",{}).get("pooled_auroc") and m.get("novel_dev",{}).get("pooled_auroc") != 0:
            return False, "missing novel AUROC"
        if not m.get("training_checkpoint_sha256"): return False, "missing checkpoint identity"
        if any("score" not in r or "label" not in r or not isinstance(r["score"],(int,float)) for r in p):
            return False, "prediction schema/score invalid"
        if any(not __import__('math').isfinite(float(r['score'])) for r in p): return False,"nonfinite score"
        # Multiple scorer arms may emit one prediction per sample. Verify unique sample coverage,
        # while allowing the same (role,qid,video,label) to carry multiple named scores.
        novel_keys={(r.get("partition_role"),str(r.get("qid")),str(r.get("vid")),int(r.get("label",-1)))
                    for r in p if r.get("partition_role")=="novel_dev"}
        if m.get("expected_novel_rows") != len(novel_keys):
            return False, "novel sample coverage mismatch"
        return True, None
    except Exception as e:
        return False, repr(e)


def run_command(task, stage, cmd, worker_id, gpu):
    log_path=Path(task["train_log"] if stage=="train" else task["eval_log"])
    log_path.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    env["RESEARCH_TASK_ID"]=task["id"]
    env["RESEARCH_GPU_ID"]=str(gpu)
    argv=[str(x) for x in cmd]
    with log_path.open("a",buffering=1) as log:
        log.write(f"\n[{now()}] START {stage}: {shlex.join(argv)}\n")
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        child_info=(child.pid,proc_start(child.pid))
        while child.poll() is None:
            heartbeat(worker_id,task["id"],child_info)
            time.sleep(5)
        log.write(f"[{now()}] EXIT {stage}: rc={child.returncode}\n")
        heartbeat(worker_id,task["id"],None)
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode,argv)


def execute_task(task, worker_id, gpu):
    spec=json.loads(task["config_json"])
    # Config carries exact argv, resolved to this task's isolated run directory.
    for path, expected in spec["code_file_hashes"].items():
        if not Path(path).is_file() or digest(Path(path)) != expected:
            with dbconn() as c:
                transition(c,task["id"],{"running"},"blocked",ended_at=now(),error=f"Code hash changed after enqueue: {path}",error_kind="provenance_mismatch")
            return
    train_cmd=spec["train_cmd"]
    eval_cmd=spec["eval_cmd"]
    try:
        if not validate_checkpoint(task):
            run_command(task,"train",train_cmd,worker_id,gpu)
        if not validate_checkpoint(task): raise RuntimeError("training exited without a valid checkpoint")
        with dbconn() as c:
            transition(c,task["id"],{"running"},"trained",trained_at=now(),checkpoint=task["checkpoint"])
            transition(c,task["id"],{"trained"},"evaluating")
        valid, reason=validate_eval(task)
        if not valid:
            run_command(task,"eval",eval_cmd,worker_id,gpu)
        valid, reason=validate_eval(task)
        if not valid: raise RuntimeError(f"evaluation artifact validation failed: {reason}")
        with dbconn() as c:
            transition(c,task["id"],{"evaluating"},"succeeded",ended_at=now(),error=None,error_kind=None)
    except BaseException as e:
        kind="oom" if "out of memory" in str(e).lower() else "command_or_validation"
        message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        with dbconn() as c:
            r=c.execute("SELECT status,attempt,max_retries FROM tasks WHERE id=?",(task["id"],)).fetchone()
            if r and r["status"] in {"running","trained","evaluating"}:
                target="ready" if r["attempt"] <= r["max_retries"] else "failed"
                transition(c,task["id"],{r["status"]},target,error=message,error_kind=kind,child_pid=None,child_start=None,
                           ended_at=now() if target=="failed" else None)
        print(f"{task['id']} {kind}: {e}",file=sys.stderr,flush=True)


def worker(gpu, idle_exit=15):
    init_db()
    recover()
    pid=os.getpid(); token=proc_start(pid)
    worker_id=f"gpu{gpu}-{pid}-{uuid.uuid4().hex[:8]}"
    with dbconn() as c:
        c.execute("BEGIN IMMEDIATE")
        occupied=c.execute("SELECT id FROM workers WHERE resource=? AND state='running'",(f"gpu:{gpu}",)).fetchone()
        if occupied: raise RuntimeError(f"GPU {gpu} already reserved by worker {occupied['id']}")
        c.execute("INSERT INTO workers(id,resource,pid,start_token,host,started_at,heartbeat_at,state) VALUES(?,?,?,?,?,?,?,'running')",
                  (worker_id,f"gpu:{gpu}",pid,token,socket.gethostname(),now(),now()))
        c.commit()
    idle_since=time.monotonic()
    while True:
        claimed=claim(str(gpu),worker_id,pid,token)
        if claimed=="busy": time.sleep(15); continue
        if claimed is None:
            with dbconn() as c: c.execute("UPDATE workers SET task_id=NULL,heartbeat_at=? WHERE id=?",(now(),worker_id))
            if time.monotonic()-idle_since>=idle_exit: break
            time.sleep(3); continue
        idle_since=time.monotonic()
        execute_task(claimed,worker_id,str(gpu))
        with dbconn() as c:
            c.execute("UPDATE workers SET task_id=NULL,heartbeat_at=? WHERE id=?",(now(),worker_id))
    with dbconn() as c: c.execute("UPDATE workers SET state='stopped',heartbeat_at=? WHERE id=?",(now(),worker_id))


def recover():
    """Recover only records whose worker identity is proven dead; never signal/kill processes."""
    init_db(); recovered=[]
    with dbconn() as c:
        c.execute("BEGIN IMMEDIATE")
        rows=c.execute("SELECT * FROM tasks WHERE status IN ('running','trained','evaluating')").fetchall()
        for row in rows:
            live=(proc_start(row["worker_pid"])==row["worker_start"] and row["worker_start"] is not None)
            task=dict(row)
            if live or task_child_is_live(task): continue
            if validate_eval(task):
                transition(c,row["id"],{row["status"]},"succeeded",ended_at=now(),error=None,error_kind=None)
                recovered.append((row["id"],"succeeded: validated existing eval"))
            elif validate_checkpoint(task) and row["status"] in ("running","trained","evaluating"):
                if row["status"]=="running":
                    transition(c,row["id"],{"running"},"trained",trained_at=now(),error="Recovered checkpoint after worker exit",error_kind="worker_interruption")
                transition(c,row["id"],{"trained","evaluating"},"ready",worker_id=None,worker_pid=None,worker_start=None,child_pid=None,child_start=None,
                           error="Recovered valid checkpoint; evaluation will resume",error_kind="worker_interruption")
                recovered.append((row["id"],"ready: checkpoint reusable"))
            else:
                target="ready" if row["attempt"]<=row["max_retries"] else "failed"
                transition(c,row["id"],{row["status"]},target,worker_id=None,worker_pid=None,worker_start=None,child_pid=None,child_start=None,
                           error="Worker exited; no validated reusable artifacts",error_kind="worker_interruption",
                           ended_at=now() if target=="failed" else None)
                recovered.append((row["id"],target+": artifacts incomplete"))
        for w in c.execute("SELECT * FROM workers WHERE state='running'").fetchall():
            if proc_start(w["pid"])!=w["start_token"]:
                c.execute("UPDATE workers SET state='stopped',heartbeat_at=? WHERE id=?",(now(),w["id"]))
        c.commit()
    for x in recovered: print(*x)


def status():
    init_db()
    with dbconn() as c:
        for r in c.execute("SELECT id,idea,backbone,fold,seed,gpu,status,attempt,heartbeat_at,error_kind FROM tasks ORDER BY created_at,id"):
            print(json.dumps(dict(r),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("init")
    e=sub.add_parser("enqueue");e.add_argument("spec")
    w=sub.add_parser("worker");w.add_argument("--gpu",required=True);w.add_argument("--idle-exit",type=int,default=15)
    sub.add_parser("recover");sub.add_parser("status")
    a=p.parse_args()
    if a.cmd=="init":init_db()
    elif a.cmd=="enqueue":enqueue(a.spec)
    elif a.cmd=="worker":worker(a.gpu,a.idle_exit)
    elif a.cmd=="recover":recover()
    elif a.cmd=="status":status()

if __name__=="__main__":main()
