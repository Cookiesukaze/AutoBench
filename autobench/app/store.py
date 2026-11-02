"""路径拼装与所有文件读写。

一次评测的全部产物都在 <data_root>/<runs_subdir>/<run_id>/ 下面，结构固定：
模块只向这里要路径，不自己拼；落盘只有这里一个出口。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .config import Config
from .types import ContractError

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ARTIFACTS = ("plan", "evidence", "quality", "judged")


class Store:
    def __init__(self, config: Config, run_id: str):
        if not RUN_ID_RE.match(run_id):
            raise ContractError(f"run_id 只能含字母数字和 ._-，得到 {run_id!r}")
        local = config.paths["local"]
        self.run_id = run_id
        self.root = Path(local["data_root"]) / local["runs_subdir"] / run_id

    # 产物路径 -------------------------------------------------------------

    def plan_file(self, question: str) -> Path:
        return self.root / "plan" / f"{question}.json"

    def evidence_dir(self, config_name: str, unit: str) -> Path:
        return self.root / "evidence" / f"{config_name}_{unit}"

    def quality_file(self, unit: str) -> Path:
        return self.root / "quality" / f"{unit}.json"

    def judged_dir(self, config_name: str, unit: str) -> Path:
        return self.root / "judged" / f"{config_name}_{unit}"

    def result_file(self, config_name: str, unit: str) -> Path:
        return self.judged_dir(config_name, unit) / "result.json"

    @property
    def cost_file(self) -> Path:
        return self.root / "cost.jsonl"

    @property
    def score_file(self) -> Path:
        return self.root / "score.json"

    @property
    def board_dir(self) -> Path:
        return self.root / "board"

    # 读写 -----------------------------------------------------------------

    def read_json(self, path: Path) -> Any:
        if not path.exists():
            raise ContractError(f"找不到 {path}；上游阶段没跑或产物被删了")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path} 不是合法 JSON：{exc}") from exc

    def write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def append_cost(self, record: dict) -> None:
        """一行一次追加，不改旧记录：重跑一个阶段只会多几行，不会覆盖历史。"""
        self.cost_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cost_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_cost(self) -> list[dict]:
        if not self.cost_file.exists():
            raise ContractError(f"还没有成本记录：{self.cost_file}")
        return [json.loads(line) for line in self.cost_file.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def judged_units(self) -> dict[str, list[str]]:
        """已经判过的 {配置: [unit]}，按目录名读出来，不猜。"""
        folder = self.root / "judged"
        if not folder.exists():
            raise ContractError(f"{folder} 不存在：judge 阶段还没跑")
        out: dict[str, list[str]] = {}
        for path in sorted(folder.iterdir()):
            if not path.is_dir() or "__seed" not in path.name:
                continue
            config_name, unit = path.name.split("_", 1)
            out.setdefault(config_name, []).append(unit)
        if not out:
            raise ContractError(f"{folder} 里没有任何判定结果")
        return out

    def units(self, config_name: str) -> list[str]:
        """这个配置已经判过的 unit，按目录名排序读出来，不猜。"""
        folder = self.root / "judged"
        if not folder.exists():
            raise ContractError(f"{folder} 不存在：judge 阶段还没跑")
        prefix = f"{config_name}_"
        return sorted(p.name[len(prefix):] for p in folder.iterdir()
                      if p.is_dir() and p.name.startswith(prefix))
