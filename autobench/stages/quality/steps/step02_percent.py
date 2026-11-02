"""专家分数的显式换算；相同原始值必须得到相同分数。"""

import math
from ....app.data.types import ContractError


def convert(values: dict, recipe: dict) -> dict:
    valid = {
        c: float(v)
        for c, v in values.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    }
    method = recipe["method"]
    if method == "percentile":
        if len(valid) < 2:
            return {c: None for c in values}
        sign = 1 if recipe["direction"] == "higher" else -1
        out = {}
        for c, v in valid.items():
            below = sum(sign * other < sign * v for other in valid.values())
            tied = sum(other == v for other in valid.values())
            out[c] = (below + (tied - 1) / 2) / (len(valid) - 1)
    elif method == "percentage":
        low, high = recipe["range"]
        if high <= low:
            raise ContractError("专家分值范围必须递增")
        if any(not low <= v <= high for v in valid.values()):
            raise ContractError("专家得分超出声明范围，不能截断或自动改用其他指标")
        out = {c: (v - low) / (high - low) for c, v in valid.items()}
        if recipe["direction"] == "lower":
            out = {c: 1 - v for c, v in out.items()}
    else:
        raise ContractError(f"未知专家评分方法：{method}")
    return {c: out.get(c) for c in values}
