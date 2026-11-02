"""读题面、校验参考图。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ....app.types import ContractError


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entries(specs: Path) -> dict[str, dict]:
    """specs 文件按 question_id 建索引。题面在 spec 或 task 字段里，两种都收。"""
    data = json.loads(specs.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "tasks" not in data:
        raise ContractError(f'{specs} 应为 {{"tasks": [...]}} 结构')
    out: dict[str, dict] = {}
    for entry in data["tasks"]:
        missing = [k for k in ("question_id", "image_sha256") if k not in entry]
        if missing:
            raise ContractError(f"{specs} 里有一条缺 {missing}")
        spec = entry.get("spec", entry.get("task"))
        if spec is None:
            # 扁平格式：条目本身就是 spec（含 segments），题号和图片哈希也在同一层。
            if "segments" not in entry:
                raise ContractError(f"{specs} 里 {entry['question_id']} 既没有 spec 字段，也不像一份 spec")
            spec = entry
        out[entry["question_id"]] = {**entry, "spec": spec}
    return out


def image_for(image_dir: Path, question: str) -> Path:
    """参考图必须按完整题号命名。命名不统一是数据的事，代码不猜。"""
    path = image_dir / f"{question}.jpg"
    if not path.exists():
        raise ContractError(f"{image_dir} 里没有 {question}.jpg；参考图要按完整题号命名")
    return path


def load(specs: Path, question: str, image_dir: Path) -> dict:
    table = entries(specs)
    if question not in table:
        raise ContractError(f"{specs} 里没有题目 {question}")
    entry = table[question]
    image = image_for(image_dir, question)
    actual = sha256(image)
    if actual != entry["image_sha256"]:
        raise ContractError(f"参考图与题面不符：{image} 的 sha256 是 {actual}")
    return {
        "question_id": question,
        "spec": entry["spec"],
        "image_path": image,
        "image_sha256": actual,
    }
