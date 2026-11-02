"""成本记录：每个阶段跑了多久、调了几次模型、花了多少 token。

只记原始量（秒、token、次数），不换算成钱——单价会变，写进代码就过期了。
记录只在 app 里产生和汇总，阶段只上报事件，不自己算账。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from .store import Store

STAGES = ("plan", "quality", "judge", "score")


class Recorder:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self._pending: dict | None = None
        self._written = 0

    def model_call(self, tokens_in: int, tokens_out: int, cached: bool) -> None:
        """一次模型调用的用量。缓存命中记 0 token，但算一次命中。"""
        if self._pending is None:
            raise RuntimeError("model_call 必须在 stage 上下文里调用")
        self._pending["calls"] += 1
        self._pending["cache_hits"] += int(cached)
        self._pending["tokens_in"] += tokens_in
        self._pending["tokens_out"] += tokens_out

    @contextmanager
    def stage(self, stage: str, unit: str, **extra) -> Iterator[None]:
        if stage not in STAGES:
            raise ValueError(f"未知阶段 {stage!r}")
        self._pending = {"calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0}
        started = time.perf_counter()
        try:
            yield
        finally:
            self.records.append({"stage": stage, "unit": unit,
                                 "wall_s": round(time.perf_counter() - started, 3),
                                 **self._pending, **extra})

    def write(self, store: Store) -> None:
        """只写还没写过的记录，重复调用不会把同一条记两遍。"""
        for record in self.records[self._written:]:
            store.append_cost(record)
        self._written = len(self.records)

    def summary(self) -> dict:
        """按阶段汇总，再给一行总计。"""
        out: dict[str, dict] = {}
        for record in self.records:
            row = out.setdefault(record["stage"], {"stages": 0, "wall_s": 0.0, "calls": 0,
                                                   "cache_hits": 0, "tokens_in": 0, "tokens_out": 0})
            row["stages"] += 1
            for key in ("wall_s", "calls", "cache_hits", "tokens_in", "tokens_out"):
                row[key] += record[key]
        out["total"] = {k: sum(r[k] for r in out.values())
                        for k in ("wall_s", "calls", "cache_hits", "tokens_in", "tokens_out")}
        return out
