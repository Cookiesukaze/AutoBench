"""SSH 与流式 tar 传输；不在内存中缓存整批视频。"""

from __future__ import annotations
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from ..data.types import ContractError


class Remote:
    def __init__(self, paths):
        self.paths = paths
        self.host = paths["host"]
        self.timeout = paths["timeouts"]

    def ssh(self, command):
        done = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f'ConnectTimeout={self.timeout["connect_s"]}',
                self.host,
                command,
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout["ssh_s"],
        )
        if done.returncode:
            raise ContractError(f"SSH 失败：{done.stderr[-800:]}")
        return done.stdout

    def upload(self, files: dict[str, Path], destination: str):
        for name in files:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts:
                raise ContractError(f"非法归档路径 {name}")
        command = f"mkdir -p {shlex.quote(destination)} && tar -C {shlex.quote(destination)} -xf -"
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", self.host, command],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=errors,
            )
            try:
                with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
                    for name, path in files.items():
                        archive.add(path, arcname=name, recursive=False)
                process.stdin.close()
                code = process.wait(timeout=self.timeout["ssh_s"])
                if code:
                    errors.seek(0)
                    raise ContractError(
                        "上传失败：" + errors.read().decode(errors="replace")[-800:]
                    )
            except BaseException:
                process.kill()
                process.wait()
                raise

    def download(self, source: str, destination: Path):
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    self.host,
                    f"tar -C {shlex.quote(source)} -cf - .",
                ],
                stdout=subprocess.PIPE,
                stderr=errors,
            )
            root = destination.resolve()
            try:
                with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                    for member in archive:
                        if (
                            member.issym()
                            or member.islnk()
                            or not (member.isfile() or member.isdir())
                        ):
                            raise ContractError(f"下载归档含非普通文件：{member.name}")
                        target = (root / member.name).resolve()
                        if not target.is_relative_to(root):
                            raise ContractError(f"下载路径越界：{member.name}")
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with archive.extractfile(member) as src, target.open(
                                "wb"
                            ) as dst:
                                shutil.copyfileobj(src, dst)
                if process.wait(timeout=self.timeout["ssh_s"]):
                    errors.seek(0)
                    raise ContractError(
                        "下载失败：" + errors.read().decode(errors="replace")[-800:]
                    )
            except BaseException:
                process.kill()
                process.wait()
                raise
