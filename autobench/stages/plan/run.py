"""阶段入口：读题面并生成本题的评测计划。"""

from __future__ import annotations

from pathlib import Path

from ...app.config import Config
from ...app.data.types import Plan
from .steps import step01_load, step02_build, step03_route


def run(specs: Path, question: str, image_dir: Path, config: Config) -> Plan:
    loaded = step01_load.load(specs, question, image_dir)
    plan = step02_build.build(loaded["spec"], loaded["question_id"],
                              loaded["image_sha256"], config.checks)
    return step03_route.route(plan, config.checks)
