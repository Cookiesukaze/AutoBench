"""批次输入与内容指纹；不通过路径猜题目或替换参考图。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..data.types import ContractError, split_unit, unit_id


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    return fingerprint(
        {
            str(p.relative_to(root)): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".md")
        }
    )


def safe_name(value: str) -> str:
    if (
        not value
        or value in (".", "..")
        or any(c in value for c in '/\\:<>"|?*')
        or any(ord(c) < 32 for c in value)
    ):
        raise ContractError(f"不安全的文件名：{value!r}")
    if value.rstrip(" .") != value or value.split(".")[0].upper() in (
        {"CON", "PRN", "AUX", "NUL"}
        | {f"COM{i}" for i in range(1, 10)}
        | {f"LPT{i}" for i in range(1, 10)}
    ):
        raise ContractError(f"Windows 不支持的文件名：{value!r}")
    return value


def discover(folder: Path) -> dict[str, dict[str, str]]:
    found = {}
    for path in sorted(Path(folder).glob("*/*/seed*/*/*.mp4")):
        config, question, seed = path.parts[-5:-2]
        match = re.fullmatch(r"seed(\d+)", seed)
        if match is None:
            continue
        unit = unit_id(question, int(match[1]))
        by = found.setdefault(config, {})
        if unit in by:
            raise ContractError(f"重复视频：{config} {unit}: {by[unit]}, {path}")
        by[unit] = str(path.resolve())
    if not found:
        raise ContractError(f"{folder} 没有 <config>/<question>/seed<N>/<run>/*.mp4")
    return found


def videos(manifest: Path | None, folder: Path | None) -> dict[str, dict[str, str]]:
    if (manifest is None) == (folder is None):
        raise ContractError("--videos 与 --videos-dir 必须且只能给一个")
    if folder is not None:
        out = discover(folder)
    else:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        out = {
            c: {u: str((manifest.parent / p).resolve()) for u, p in by.items()}
            for c, by in raw.items()
        }
    if not out or any(not by for by in out.values()):
        raise ContractError("批次或配置的视频清单为空")
    for c, by in out.items():
        safe_name(c)
        for u, path in by.items():
            safe_name(u)
            split_unit(u)
            if not Path(path).is_file():
                raise ContractError(f"视频不存在：{path}")
    return out
