"""读四个 json 并校验，缺键直接抛。

这一层不提供默认值。配置里没写的键就是没写，调用方必须显式处理，
不许在这里悄悄补一个数让流程跑下去。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ContractError, JUDGED_BY

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
FILES = ("checks", "vlm", "paths", "gpus")

AGGREGATIONS = ("dimension", "item")
DIRECTIONS = ("higher", "lower")
DETAILS = ("low", "high", "auto")


@dataclass(frozen=True)
class Config:
    checks: dict[str, Any]
    vlm: dict[str, Any]
    paths: dict[str, Any]
    gpus: dict[str, Any]


def load(config_dir: Path = CONFIG_DIR) -> Config:
    unknown = sorted(set(p.stem for p in config_dir.glob("*.json")) - set(FILES))
    if unknown:
        raise ContractError(f"{config_dir} 里有认不出的配置文件：{unknown}")
    data = {name: _read(config_dir / f"{name}.json") for name in FILES}
    _validate_checks(data["checks"])
    _validate_vlm(data["vlm"])
    _validate_paths(data["paths"])
    _validate_gpus(data["gpus"])
    _validate_cross(data)
    return Config(**data)


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(f"缺少配置文件 {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path} 不是合法 JSON：{exc}") from exc


def _need(where: str, block: Any, keys: tuple[str, ...]) -> None:
    if not isinstance(block, dict):
        raise ContractError(f"{where} 应为对象，得到 {type(block).__name__}")
    missing = [k for k in keys if k not in block]
    if missing:
        raise ContractError(f"{where} 缺字段 {missing}")


def _number(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} 应为数字，得到 {value!r}")
    return float(value)


def _validate_checks(d: dict[str, Any]) -> None:
    _need("checks.json", d, ("checks", "experts", "weights", "thresholds"))
    experts = d["experts"]
    categories = d["weights"].get("categories") or {}
    if not categories:
        raise ContractError("weights.categories 为空：维度前缀无处归类")

    for name, item in d["checks"].items():
        where = f"checks.checks.{name}"
        _need(where, item, ("dimension", "judged_by"))
        if item["judged_by"] not in JUDGED_BY:
            raise ContractError(f"{where}.judged_by 只能是 {JUDGED_BY}")
        prefix = str(item["dimension"]).split("-")[0]
        if prefix not in categories:
            raise ContractError(f"{where}.dimension={item['dimension']} 的前缀不在 weights.categories 里")
        expert = item.get("expert")
        if expert is not None and expert not in experts:
            raise ContractError(f"{where}.expert={expert!r} 没有在 experts 里定义")

    for name, spec in experts.items():
        _need(f"checks.experts.{name}", spec, ("installed",))
        if not isinstance(spec["installed"], bool):
            raise ContractError(f"checks.experts.{name}.installed 应为布尔")

    w = d["weights"]
    _need("weights", w, ("content", "physics", "quality", "aggregation", "min_coverage"))
    for key in ("content", "physics", "quality"):
        if _number(f"weights.{key}", w[key]) < 0:
            raise ContractError(f"weights.{key} 不能为负")
    if w["aggregation"] not in AGGREGATIONS:
        raise ContractError(f"weights.aggregation 只能是 {AGGREGATIONS}")
    if not 0.0 <= _number("weights.min_coverage", w["min_coverage"]) <= 1.0:
        raise ContractError("weights.min_coverage 应在 [0,1]")

    t = d["thresholds"]
    _need("thresholds", t, ("quality", "judge"))
    _need("thresholds.quality", t["quality"], ("scale", "detectors"))
    t = {"scale": t["quality"]["scale"], "detectors": t["quality"]["detectors"],
         "judge": t["judge"]}
    _need("thresholds.scale", t["scale"],
          ("screen_score", "danger_score", "score_center_z", "parent_z", "mad_to_sigma",
           "min_configs", "bad_share", "config_threshold"))
    if _number("thresholds.quality.scale.danger_score", t["scale"]["danger_score"]) <= \
            _number("thresholds.quality.scale.screen_score", t["scale"]["screen_score"]):
        raise ContractError("thresholds.quality.scale.danger_score 必须高于 screen_score")

    for name, spec in t["detectors"].items():
        where = f"thresholds.quality.detectors.{name}"
        _need(where, spec, ("worse_when",))
        if spec["worse_when"] not in DIRECTIONS:
            raise ContractError(f"{where}.worse_when 只能是 {DIRECTIONS}")
        if "p" in spec:
            if not 0.0 <= _number(f"{where}.p", spec["p"]) <= 1.0:
                raise ContractError(f"{where}.p 应在 [0,1]")
        elif spec.get("counts") is not False:
            raise ContractError(f"{where} 参与扣分却没有 p")
        if "group_z" in spec:
            _number(f"{where}.group_z", spec["group_z"])
        if spec.get("absolute") is not None and not isinstance(spec["absolute"], (int, float, dict)):
            raise ContractError(f"{where}.absolute 应为数字、对象或省略")

    _need("thresholds.judge", t["judge"], ("size_outlier_ratio", "hand_score", "hand_reach"))


def _validate_vlm(d: dict[str, Any]) -> None:
    _need("vlm.json", d, ("model", "effort", "max_output_tokens", "image", "frames",
                          "timeout", "retry", "vote", "concurrency", "endpoint", "budget"))
    _need("vlm.image", d["image"], ("max_side", "jpeg_quality", "detail", "max_per_request"))
    if d["image"]["detail"] not in DETAILS:
        raise ContractError(f"vlm.image.detail 只能是 {DETAILS}")
    if int(d["image"]["max_per_request"]) < 1:
        raise ContractError("vlm.image.max_per_request 应至少为 1")
    _need("vlm.frames", d["frames"], ("locate", "window", "review", "review_margin_s",
                                      "event_margin_s", "end_window_s", "jpeg_quality",
                                      "stagger_votes", "sheet", "event_tail_to", "review_sheets"))
    if d["frames"]["event_tail_to"] not in ("margin", "next", "end"):
        raise ContractError("vlm.frames.event_tail_to 只能是 margin / next / end")
    if not isinstance(d["frames"]["stagger_votes"], bool):
        raise ContractError("vlm.frames.stagger_votes 只能是 true 或 false")
    if int(d["frames"]["review_sheets"]) < 1:
        raise ContractError("vlm.frames.review_sheets 至少为 1")
    _need("vlm.frames.sheet", d["frames"]["sheet"], ("cols", "rows", "side", "gap", "apply_to"))
    for key in ("cols", "rows", "side", "gap"):
        if int(d["frames"]["sheet"][key]) < 1:
            raise ContractError(f"vlm.frames.sheet.{key} 至少为 1")
    stages = d["frames"]["sheet"]["apply_to"]
    if not isinstance(stages, list) or any(s not in ("locate", "review") for s in stages):
        raise ContractError("vlm.frames.sheet.apply_to 只能是 locate / review")
    _need("vlm.timeout", d["timeout"], ("connect_s", "read_s"))
    _need("vlm.retry", d["retry"], ("transport_attempts", "format_attempts", "stage"))
    for stage in ("first", "fusion", "review"):
        _need(f"vlm.retry.stage.{stage}", d["retry"]["stage"].get(stage, {}),
              ("attempts", "transport_sleep_s", "format_sleep_s"))
    _need("vlm.vote", d["vote"], ("first", "fusion", "review", "max_votes"))
    for stage in ("first", "fusion", "review"):
        if int(d["vote"][stage]) < 1:
            raise ContractError(f"vlm.vote.{stage} 至少为 1（1 就是只问一次）")
    if int(d["vote"]["max_votes"]) < max(int(d["vote"][s]) for s in ("first", "fusion", "review")):
        raise ContractError("vlm.vote.max_votes 不能小于任何阶段的票数")
    if int(d["concurrency"]) < 1:
        raise ContractError("vlm.concurrency 至少为 1")
    _need("vlm.budget", d["budget"], ("max_calls_per_video", "max_review_requests"))
    if int(d["budget"]["max_review_requests"]) < 1:
        raise ContractError("vlm.budget.max_review_requests 至少为 1")
    if int(d["budget"]["max_calls_per_video"]) < 2:
        raise ContractError("vlm.budget.max_calls_per_video 至少为 2（第一问+第二问）")
    _need("vlm.endpoint", d["endpoint"], ("source", "base_url", "api_key_env"))
    source = d["endpoint"]["source"]
    if source not in ("env", "cc-switch"):
        raise ContractError("vlm.endpoint.source 只能是 env 或 cc-switch")
    if source == "cc-switch" and not d["endpoint"].get("cc_switch_db"):
        raise ContractError("vlm.endpoint.source=cc-switch 时必须给 cc_switch_db")
    if source == "env" and not d["endpoint"].get("api_key_env"):
        raise ContractError("vlm.endpoint.source=env 时必须给 api_key_env")


def _validate_paths(d: dict[str, Any]) -> None:
    _need("paths.json", d, ("local", "remote"))
    _need("paths.local", d["local"], ("data_root", "runs_subdir"))
    if not d["local"]["data_root"]:
        raise ContractError("paths.local.data_root 为空")
    remote = d["remote"]
    _need("paths.remote", remote, ("host", "base", "repo", "third_party", "expert_weights",
                                   "expert_manifests", "manifests", "python",
                                   "expert_python", "assets", "archive", "timeouts"))
    if not str(remote["base"]).startswith("/"):
        raise ContractError("paths.remote.base 应为绝对路径")
    for name, interpreter in remote["expert_python"].items():
        if interpreter not in remote["python"]:
            raise ContractError(f"paths.remote.expert_python.{name}={interpreter!r} 不在 python 里")
    _need("paths.remote.timeouts", remote["timeouts"],
          ("ssh_s", "connect_s", "start_s", "start_retries", "start_sleep_s", "poll_s", "poll_limit_s"))


def _validate_gpus(d: dict[str, Any]) -> None:
    _need("gpus.json", d, ("count", "free_mib", "headroom_gib", "max_sam_servers", "budget_gib"))
    if int(d["count"]) < 1:
        raise ContractError("gpus.count 至少为 1")
    if not d["budget_gib"]:
        raise ContractError("gpus.budget_gib 为空")
    for step, gib in d["budget_gib"].items():
        if _number(f"gpus.budget_gib.{step}", gib) <= 0:
            raise ContractError(f"gpus.budget_gib.{step} 应大于 0")


def _validate_cross(d: dict[str, Any]) -> None:
    """四个文件之间的一致性。分开看都合法、合起来矛盾的情况在这里拦下。"""
    experts = set(d["checks"]["experts"])
    missing = sorted(experts - set(d["gpus"]["budget_gib"]))
    if missing:
        raise ContractError(f"专家 {missing} 没有在 gpus.budget_gib 里给出显存预算")
    used = {expert for item in d["checks"]["checks"].values() if (expert := item.get("expert"))}
    missing = sorted(used - experts)
    if missing:
        raise ContractError(f"检查项引用的专家 {missing} 没有在 experts 里定义")
