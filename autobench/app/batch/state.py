"""批次独占、任务状态与可审计的精确续跑。"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ..data.types import ContractError
from .inputs import sha256


@contextmanager
def exclusive(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "batch.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ContractError(
            f"{lock} 已存在；核对原进程退出后再移除锁，禁止并发写同一批次"
        ) from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


class State:
    def __init__(self, store):
        self.store = store
        self.path = store.root / "batch.json"
        self.data = (
            store.read_json(self.path)
            if self.path.exists()
            else {"tasks": {}, "history": []}
        )
        self.lock = threading.Lock()

    def reusable(self, key: str, identity: str, artifacts: list[Path]) -> bool:
        old = self.data["tasks"].get(key) or {}
        return (
            old.get("status") == "done"
            and old.get("identity") == identity
            and all(
                p.is_file() and old.get("artifacts", {}).get(str(p)) == sha256(p)
                for p in artifacts
            )
        )

    def record(self, key: str, identity: str, status: str, artifacts=(), error=None):
        with self.lock:
            row = {
                "key": key,
                "identity": identity,
                "status": status,
                "time": time.time(),
                "error": error,
                "artifacts": {str(p): sha256(p) for p in artifacts if p.is_file()},
            }
            self.data["tasks"][key] = row
            self.data["history"].append(row)
            self.store.write_json(self.path, self.data)
