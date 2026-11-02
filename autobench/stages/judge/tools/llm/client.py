"""和模型通话：发请求、按请求内容缓存、传输失败重试。

地址与密钥从环境变量读，不写进仓库。缺了就报错，不猜默认端点。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .....app.data.types import ContractError

BASE_URL_ENV = "AUTOBENCH_VLM_BASE_URL"
CC_SWITCH_URL = re.compile(r'^\s*base_url\s*=\s*"([^"]+)"', re.M)
BARE_KEY = re.compile(r'([{,])[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*:')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        blob = match.group(0)
        quoted = BARE_KEY.sub(lambda m: m.group(1) + '"' + m.group(2) + '":', blob)
        for candidate in (blob, quoted):
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("Incomplete VLM response")


def _save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _encode(path: Path, max_side: int, quality: int, detail: str) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _credentials(endpoint: dict) -> tuple[str, str]:
    """按配置里写明的来源取地址和密钥。两种来源都显式，不做其中一种失败就偷用另一种。"""
    source = endpoint["source"]
    if source == "env":
        base_url = endpoint.get("base_url") or os.environ.get(BASE_URL_ENV)
        if not base_url:
            raise ContractError(f"没有接口地址：填 vlm.endpoint.base_url，或设置 {BASE_URL_ENV}")
        key = os.environ.get(endpoint["api_key_env"])
        if not key:
            raise ContractError(f"没有密钥：环境变量 {endpoint['api_key_env']} 未设置")
        return str(base_url).rstrip("/"), key
    if source == "cc-switch":
        db = Path(os.path.expanduser(endpoint["cc_switch_db"]))
        if not db.exists():
            raise ContractError(f"凭据库不存在：{db}")
        import sqlite3
        with sqlite3.connect("file:" + db.as_posix() + "?mode=ro", uri=True) as con:
            rows = con.execute("SELECT id, settings_config FROM providers "
                               "WHERE app_type='codex' AND is_current=1").fetchall()
        if len(rows) != 1:
            raise ContractError(f"{db} 里当前 codex provider 有 {len(rows)} 个，必须正好 1 个")
        provider, raw = rows[0]
        settings = json.loads(raw)
        urls = CC_SWITCH_URL.findall(settings.get("config", ""))
        if len(urls) != 1:
            raise ContractError(f"provider {provider} 的 base_url 有 {len(urls)} 个，必须正好 1 个")
        key = (settings.get("auth") or {}).get("OPENAI_API_KEY")
        if not key:
            raise ContractError(f"provider {provider} 没有 OPENAI_API_KEY")
        return urls[0].rstrip("/"), key
    raise ContractError(f"不认识的凭据来源 {source!r}")


def _cap(frames: list[dict], limit: int) -> list[dict]:
    """超过上限时按等间隔抽稀，并保住最后一帧——结尾状态只能靠它读。"""
    if len(frames) <= limit:
        return frames
    rest, last = frames[:-1], frames[-1]
    step = len(rest) / (limit - 1)
    return [last] + [rest[int(i * step)] for i in range(limit - 1)]


class Client:
    def __init__(self, vlm: dict, cache_dir: Path, recorder=None):
        self.url, self.key = _credentials(vlm["endpoint"])
        self.vlm = vlm
        self.recorder = recorder
        self.cache = Path(cache_dir) / "calls"
        self.limit = int(vlm["budget"]["max_calls_per_video"])
        self.calls = 0

    def _spend(self) -> None:
        """一条视频的调用次数是硬预算。超了就停机，不许悄悄多问。"""
        self.calls += 1
        if self.calls > self.limit:
            raise ContractError("这条视频的模型调用次数超过预算 %d 次" % self.limit)

    def ask(self, name: str, prompt: str, frames: list[dict]) -> dict:
        """frames[0] 是参考图，其余是视频帧。返回解析后的 JSON。"""
        payload = self._payload(prompt, frames)
        identity = digest(json.dumps(payload, sort_keys=True).encode())[:16]
        folder = self.cache / f"{name}_{identity}"
        cached = folder / "parsed.json"
        if cached.exists():
            if self.recorder is not None:
                self.recorder.model_call(0, 0, cached=True)
            return json.loads(cached.read_text(encoding="utf-8"))

        retry = self.vlm["retry"]
        transport_tries = format_tries = 0
        while True:
            try:
                parsed = self._once(folder, payload)
                self._spend()
                _save(cached, parsed)
                return parsed
            except (requests.RequestException, RuntimeError) as exc:
                if isinstance(exc, requests.RequestException):
                    transport_tries += 1
                    if transport_tries >= retry["transport_attempts"]:
                        raise
                    low, high = retry["transport_backoff_s"]
                    delay = min(high, low * 2 ** (transport_tries - 1))
                else:
                    format_tries += 1
                    if format_tries >= retry["format_attempts"]:
                        raise
                    delay = retry["stage"]["first"]["format_sleep_s"]
                _save(folder / f"retry_{transport_tries}_{format_tries}.json",
                      {"error": str(exc), "type": type(exc).__name__, "same_request": True})
                time.sleep(delay)

    def ask_validated(self, name: str, prompt: str, frames: list[dict], validate,
                      stage: str) -> dict:
        """问一次并校验。校验不过就把错误原样贴回去重问，轮数用完仍然不过就抛。"""
        plan = self.vlm["retry"]["stage"][stage]
        attempt_prompt = prompt
        for attempt in range(plan["attempts"]):
            out = self.ask(name, attempt_prompt, frames)
            try:
                validate(out)
                return out
            except ValueError as exc:
                if attempt == plan["attempts"] - 1:
                    raise
                attempt_prompt = (prompt + "\n\n你上一次的回答有格式错误，必须修正后重新回答全部条目：\n"
                                  + str(exc))
                time.sleep(plan["format_sleep_s"])
        raise RuntimeError("unreachable")

    def _payload(self, prompt: str, frames: list[dict]) -> dict:
        image_cfg = self.vlm["image"]
        capped = _cap(frames, image_cfg["max_per_request"])
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for frame in capped:
            content.append({"type": "input_text",
                            "text": json.dumps({k: v for k, v in frame.items() if k != "path"},
                                               ensure_ascii=False)})
            content.append({"type": "input_image", "detail": image_cfg["detail"],
                            "image_url": _encode(Path(frame["path"]), image_cfg["max_side"],
                                                 image_cfg["jpeg_quality"], image_cfg["detail"])})
        return {"model": self.vlm["model"],
                "reasoning": {"effort": self.vlm["effort"]},
                "input": [{"role": "user", "content": content}],
                "store": False,
                "max_output_tokens": self.vlm["max_output_tokens"],
                "text": {"format": {"type": "json_object"}}}

    def _once(self, folder: Path, payload: dict) -> dict:
        folder.mkdir(parents=True, exist_ok=True)
        timeout = (self.vlm["timeout"]["connect_s"], self.vlm["timeout"]["read_s"])
        started = time.time()
        with requests.Session() as session:
            session.trust_env = False
            with session.post(self.url + "/responses", json={**payload, "stream": True},
                              stream=True, headers={"Authorization": "Bearer " + self.key},
                              timeout=timeout) as response:
                if not response.ok:
                    raise RuntimeError("VLM HTTP " + str(response.status_code))
                result = None
                with (folder / "events.jsonl").open("w", encoding="utf-8") as log:
                    for line in response.iter_lines():
                        if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                            continue
                        event = json.loads(line[6:])
                        log.write(json.dumps(event, ensure_ascii=False) + "\n")
                        log.flush()
                        if event.get("type") == "response.completed":
                            result = event["response"]
                            break
                        if event.get("type") in ("error", "response.failed", "response.incomplete"):
                            raise RuntimeError(event["type"])
        if result is None or result.get("status") != "completed":
            raise RuntimeError("Incomplete VLM response")
        _save(folder / "response.json", result)
        text = "".join(c["text"] for o in result["output"] if o["type"] == "message"
                       for c in o["content"] if c["type"] == "output_text")
        usage = result.get("usage") or {}
        _save(folder / "metrics.json", {"elapsed_s": time.time() - started, "usage": usage,
                                        "returned_model": result.get("model")})
        if self.recorder is not None:
            self.recorder.model_call(int(usage.get("input_tokens", 0)),
                                     int(usage.get("output_tokens", 0)), cached=False)
        return _parse_json(text)
