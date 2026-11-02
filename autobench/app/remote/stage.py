"""隔离代码快照、视频上传、远端专家作业和校验回收。"""

from __future__ import annotations
import json
import shlex
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from ..data.store import Store
from ..data.types import ContractError
from .ssh import Remote
from .collect import verify, promote


def run(store, config, tasks, identity):
    paths = config.paths["remote"]
    remote = Remote(paths)
    root = str(
        PurePosixPath(paths["base"])
        / paths["archive"]
        / "autobench"
        / store.run_id
        / identity
    )
    code = Path(__file__).resolve().parents[2]
    files = {
        "code/autobench/" + p.relative_to(code).as_posix(): p
        for p in code.rglob("*")
        if p.is_file() and p.suffix in (".py", ".md")
    }
    remote_tasks = []
    for task in tasks:
        video_key = "videos/" + task["video_sha256"] + ".mp4"
        files[video_key] = Path(task["video"])
        remote_tasks.append(dict(task, video=root + "/" + video_key))
    job = dict(
        tasks=remote_tasks,
        remote=paths,
        gpu_config=config.gpus,
        checks=config.checks,
        gpus=list(range(config.gpus["count"])),
        identity=identity,
    )
    with tempfile.TemporaryDirectory() as tmp:
        job_file = Path(tmp) / "job.json"
        Store.write_json(job_file, job)
        files["job.json"] = job_file
        exists = (
            remote.ssh(
                "test -f " + shlex.quote(root + "/READY") + " && echo yes || true"
            ).strip()
            == "yes"
        )
        if not exists:
            remote.upload(files, root)
            manifests = str(PurePosixPath(paths["base"]) / paths["expert_manifests"])
            commands = ["mkdir -p " + shlex.quote(root + "/manifests")]
            for expert, relative in paths["manifests"].items():
                commands.append(
                    "cp "
                    + shlex.quote(manifests + "/" + relative)
                    + " "
                    + shlex.quote(root + "/manifests/" + expert + ".json")
                )
            remote.ssh(" && ".join(commands))
            remote.ssh("touch " + shlex.quote(root + "/READY"))
    python = str(PurePosixPath(paths["base"]) / paths["python"]["tracker"])
    launcher = """import os,json,sys,traceback
from pathlib import Path
p=Path(sys.argv[1]);lock=os.fdopen(int(sys.argv[3]),'a')
s=p/'out/status.json'
done=s.exists() and json.loads(s.read_text()).get('stage')=='DONE' and (p/'inventory.json').exists()
if done: sys.exit(0)
try:
    from autobench.app.remote.gpu.runner import run_job
    run_job(p/'job.json')
except BaseException:
    s.write_text(json.dumps(dict(stage='FAILED',error=traceback.format_exc())))
    raise
"""
    # Spawn through Python so SSH closes promptly and the process owns its session.
    token = uuid.uuid4().hex
    spawn = """import fcntl,json,subprocess,sys
from pathlib import Path
p=Path(sys.argv[1]); lock=(p/'runner.lock').open('a')
try:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    print((p/'attempt.txt').read_text());sys.exit(0)
token=sys.argv[3];(p/'out').mkdir(exist_ok=True)
status=p/'out/status.json'
done=status.exists() and json.loads(status.read_text()).get('stage')=='DONE' and (p/'inventory.json').exists()
if not done: status.write_text(json.dumps({'stage':'starting'}))
(p/'attempt.txt').write_text(token)
log=(p/'runner.log').open('a')
subprocess.Popen([sys.executable,'-c',sys.argv[2],str(p),token,str(lock.fileno())],
    cwd=p/'code',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,pass_fds=(lock.fileno(),))
print(token)
"""
    token = remote.ssh(
        shlex.quote(python)
        + " -c "
        + shlex.quote(spawn)
        + " "
        + shlex.quote(root)
        + " "
        + shlex.quote(launcher)
        + " "
        + shlex.quote(token)
    ).strip()
    started = time.monotonic()
    while time.monotonic() - started < paths["timeouts"]["poll_limit_s"]:
        acknowledged = remote.ssh(
            "cat " + shlex.quote(root + "/attempt.txt") + " 2>/dev/null || true"
        ).strip()
        raw = remote.ssh(
            "cat " + shlex.quote(root + "/out/status.json") + " 2>/dev/null || true"
        ).strip()
        if raw:
            status = json.loads(raw)
            if status["stage"] == "FAILED":
                raise ContractError("远端专家失败：" + status.get("error", "")[-2000:])
            if (
                status["stage"] == "DONE"
                or status["stage"] == "PARTIAL"
                and acknowledged == token
            ):
                inventory = remote.ssh(
                    "cat "
                    + shlex.quote(root + "/inventory.json")
                    + " 2>/dev/null || true"
                ).strip()
                if inventory:
                    destination = store.root / "remote" / identity
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(
                        dir=destination.parent, prefix="collect-"
                    ) as tmp:
                        remote.download(root + "/out", Path(tmp))
                        verify(Path(tmp), json.loads(inventory))
                        promote(Path(tmp), destination)
                    Store.write_json(
                        destination / "inventory.json", json.loads(inventory)
                    )
                    return destination, status
        print(
            "remote: 专家作业运行中，已等待 "
            + str(round(time.monotonic() - started))
            + " 秒",
            flush=True,
        )
        time.sleep(min(30, paths["timeouts"]["poll_s"]))
    raise ContractError(
        "远端轮询超时；远端作业保留，使用同一批次续跑以重新连接：" + root
    )
