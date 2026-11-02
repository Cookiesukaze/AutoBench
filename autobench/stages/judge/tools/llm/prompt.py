"""提示词渲染：从 autobench/prompt/ 读 md，填 $占位符，填不上就报错。"""

from __future__ import annotations

from pathlib import Path
from string import Template

from .....app.data.types import ContractError

PROMPT_DIR = Path(__file__).resolve().parents[4] / "prompt"


def render(name: str, **values: object) -> str:
    """name 形如 judge/first，对应 autobench/prompt/judge/first.md。"""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise ContractError(f"没有提示词 {path}")
    try:
        return Template(path.read_text(encoding="utf-8")).substitute(**values)
    except KeyError as exc:
        raise ContractError(f"{path} 里有未提供的占位符 {exc}") from exc
    except ValueError as exc:
        raise ContractError(f"{path} 的占位符写法有问题：{exc}") from exc
