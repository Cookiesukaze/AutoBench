"""数据契约：Plan / Item / Flag / Verdict / Score，以及它们的不变量。

跨模块只传这里定义的对象，不传裸 dict。字段增删只在这里发生。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

JUDGED_BY = ("vlm", "rule")

UNIT_RE = re.compile(r"^(?P<question>.+)__seed(?P<seed>\d+)$")


class ContractError(Exception):
    """契约被违反。宁可在这里炸，也不带着错数据往下跑。"""


def unit_id(question: str, seed: int) -> str:
    """一个题目加一个种子的评测单元，例如 B026_E1__seed48。"""
    return f"{question}__seed{int(seed)}"


def split_unit(unit: str) -> tuple[str, int]:
    match = UNIT_RE.match(unit)
    if match is None:
        raise ContractError(f"unit 格式应为 <question>__seed<N>，得到 {unit!r}")
    return match["question"], int(match["seed"])


class Outcome(str, Enum):
    """一个检查项只有三种结局，任何两种之间不许互相冒充。"""

    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class Item:
    id: str
    type: str
    dimension: str
    judged_by: str
    expert: str | None
    standard: str
    rule: dict[str, Any] = field(default_factory=dict)
    levels: tuple[float, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    concepts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.judged_by not in JUDGED_BY:
            raise ContractError(f"{self.id}: judged_by 只能是 {JUDGED_BY}，得到 {self.judged_by!r}")
        if self.judged_by == "rule" and not self.expert:
            raise ContractError(f"{self.id}: 规则判定的项必须写明用哪个专家")
        if self.levels and 1.0 not in self.levels:
            raise ContractError(f"{self.id}: 分档 {self.levels} 里没有满分 1")


@dataclass(frozen=True)
class Event:
    id: str
    what: str
    completion: dict[str, str]
    dimension: str
    constraints: tuple[Item, ...] = ()
    physics: tuple[Item, ...] = ()
    concepts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = {"full", "partial", "none"} - set(self.completion)
        if missing:
            raise ContractError(f"事件 {self.id} 的完成标准缺 {sorted(missing)}")


@dataclass(frozen=True)
class Plan:
    question_id: str
    prompt: str
    reference_sha256: str
    events: tuple[Event, ...]
    items: tuple[Item, ...] = ()
    subject_descriptions: dict[str, str] = field(default_factory=dict)

    def all_items(self) -> Iterator[Item]:
        for event in self.events:
            yield from event.constraints
            yield from event.physics
        yield from self.items

    def item(self, item_id: str) -> Item:
        for candidate in self.all_items():
            if candidate.id == item_id:
                return candidate
        raise ContractError(f"{self.question_id}: 计划里没有 {item_id}")


@dataclass(frozen=True)
class Flag:
    """一个检测器对一条视频的结论。flagged=None 表示没测到，不是干净。"""

    detector: str
    flagged: bool | None
    routes: dict[str, Any] = field(default_factory=dict)
    suspect: bool = False

    def __post_init__(self) -> None:
        if self.flagged is None and not self.routes:
            raise ContractError(f"检测器 {self.detector} 报未测量时必须留下原因")


@dataclass(frozen=True)
class Verdict:
    """一个检查项的判定。MEASURED 必须有值，另外两种必须没有值。"""

    id: str
    outcome: Outcome
    value: float | None = None
    reason: str = ""
    step: str = ""
    votes: dict[str, int] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome is Outcome.MEASURED:
            if self.value is None:
                raise ContractError(f"{self.id}: 判定为已测却没有数值")
            if not 0.0 <= self.value <= 1.0:
                raise ContractError(f"{self.id}: 分值 {self.value} 超出 [0,1]")
        elif self.value is not None:
            raise ContractError(f"{self.id}: 判定为 {self.outcome.value} 却带着数值 {self.value}")


@dataclass(frozen=True)
class Score:
    """一条视频的三个分数。任何一维没定，总分就没定——不拿别的东西填。"""

    instruction: float | None
    physics: float | None
    quality: float | None
    coverage: float
    by_dimension: dict[str, float] = field(default_factory=dict)

    def total(self, weights: dict[str, float]) -> float:
        parts = {
            "instruction": self.instruction,
            "physics": self.physics,
            "quality": self.quality,
        }
        missing = [name for name, value in parts.items() if value is None]
        if missing:
            raise ContractError(f"总分未定，缺 {missing}")
        used = {name: weights[name] for name in parts if weights.get(name, 0) > 0}
        if not used:
            raise ContractError("三个维度的权重全为 0，算不出总分")
        return sum(used[name] * parts[name] for name in used) / sum(used.values())
