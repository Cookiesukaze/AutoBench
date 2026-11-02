"""合票：多数决，平局再抽一张，抽到上限还平就认未定。

单模型多采样、将来多判官各投一票，都走这一处：先把票凑够，再看最高票是否并列。
并列时不许自己挑一个——要么继续抽，要么记未定。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable


def counts(values: list[Any]) -> dict[Any, int]:
    return dict(Counter(values))


def tied(values: list[Any]) -> bool:
    """最高票和别人并列。一张票都没有也算并列：没有结论。"""
    top = Counter(values).most_common()
    if not top:
        return True
    return len(top) > 1 and top[1][1] == top[0][1]


def collect(draw: Callable[[int], Any], samples: int, max_votes: int,
            still_tied: Callable[[list], bool]) -> list:
    """先抽 samples 张；只要 still_tied 说还分不出胜负，就继续抽，最多 max_votes 张。"""
    sheets = [draw(i) for i in range(samples)]
    while len(sheets) < max_votes and still_tied(sheets):
        sheets.append(draw(len(sheets)))
    return sheets


def winner(pairs: list[tuple[Any, Any]]) -> tuple[Any, dict, bool]:
    """pairs: [(票面, 那一票的原文)] → (胜出票的原文, 票数, 是否全票一致)。

    并列第一时返回 (None, 票数, False)；调用方拿到 None 就该记未定，不许代选。
    """
    top = Counter(value for value, _ in pairs).most_common()
    if len(top) > 1 and top[1][1] == top[0][1]:
        return None, dict(top), False
    value = top[0][0]
    chosen = next(row for face, row in pairs if face == value)
    return chosen, dict(top), len(top) == 1
