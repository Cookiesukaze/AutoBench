"""结尾三档的闭式追问。

模型有时不用 1 / 0.9 / 0.5，而是回一个 pass 或 resolved。这时不再让它描述，
直接问一个只有三个答案的题，把它逼回档位上。
"""

from __future__ import annotations

from pathlib import Path

from .....app.types import Item

LEVELS = (1.0, 0.9, 0.5)


def ask(client, item: Item, reference: Path, frames: list[dict]) -> tuple[float, str] | None:
    prompt = ('只回答一个 JSON 对象，不要任何其它文字：{"level": 数字}\n\n'
              "判定标准：" + item.standard + "\n"
              "1 = 结尾完全达到要求；0.9 = 结尾正朝要求的方向进行、但还没完全到位；"
              "0.5 = 画面里完全没有要求的结尾。")
    try:
        out = client.ask("end_state_level_" + item.id.replace(".", "_"), prompt,
                         [{"id": "reference", "path": str(reference)}] + frames)
    except Exception:
        return None
    try:
        level = float(out.get("level") if isinstance(out, dict) else out)
    except (TypeError, ValueError):
        return None
    return min(LEVELS, key=lambda x: abs(x - level)), f"追问回答 {level!r}"
