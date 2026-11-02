"""本地与远端批次评测入口。"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import run
from .config import load, CONFIG_DIR
from .data.store import Recorder, Store


def main(argv=None):
    parser = argparse.ArgumentParser(prog="autobench")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-id", required=True)
    common.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    p = sub.add_parser("plan", parents=[common])
    p.add_argument("--specs", type=Path, required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    for name in ("basic-quality", "quality"):
        q = sub.add_parser(name, parents=[common])
        q.add_argument("--quality-dir", type=Path, required=True)
        q.add_argument("--units", nargs="+", required=True)
        if name == "basic-quality":
            q.add_argument("--parents", type=Path)
    j = sub.add_parser("judge", parents=[common])
    j.add_argument("--config", required=True)
    j.add_argument("--unit", required=True)
    j.add_argument("--video", type=Path, required=True)
    j.add_argument("--image", type=Path, required=True)
    j.add_argument("--experts", type=Path, required=True)
    sub.add_parser("score", parents=[common])
    b = sub.add_parser("batch", parents=[common])
    b.add_argument("--specs", type=Path, required=True)
    b.add_argument("--image-dir", type=Path, required=True)
    source = b.add_mutually_exclusive_group(required=True)
    source.add_argument("--videos", type=Path)
    source.add_argument("--videos-dir", type=Path)
    b.add_argument("--parents", type=Path, help="JSON: 配置名到原版配置名")
    b.add_argument("--quality-dir", type=Path)
    b.add_argument("--experts-dir", type=Path)
    b.add_argument("--remote", action="store_true", help="上传并运行远端 GPU 专家")
    b.add_argument(
        "--judge", action="store_true", help="继续正式 VLM 判定和计分；会产生 API 请求"
    )
    a = parser.parse_args(argv)
    config = load(a.config_dir)
    store = Store(config, a.run_id)
    rec = Recorder()
    parents = (
        json.loads(a.parents.read_text(encoding="utf-8"))
        if getattr(a, "parents", None)
        else None
    )
    if a.command == "plan":
        result = run.plan_one(store, config, rec, a.specs, a.question, a.image_dir)
        print("plan:", result.question_id)
    elif a.command == "judge":
        result = run.judge_one(
            store, config, rec, a.config, a.unit, a.video, a.image, a.experts
        )
        print("judge:", a.config, a.unit)
    elif a.command == "score":
        result = run.score_all(store, config, rec)
        print("ranked:", result["board"]["ranked_order"])
    elif a.command == "basic-quality":
        result = run.basic_quality(store, config, rec, a.quality_dir, a.units, parents)
        print("basic-quality:", result["rates"])
    elif a.command == "quality":
        result = run.quality(store, config, rec, a.quality_dir, a.units)
        print("quality:", result["scores"])
    else:
        from .batch.run import run as batch

        result = batch(
            store,
            config,
            specs=a.specs,
            image_dir=a.image_dir,
            manifest=a.videos,
            videos_dir=a.videos_dir,
            parents=parents,
            quality_dir=a.quality_dir,
            experts_dir=a.experts_dir,
            remote=a.remote,
            judge=a.judge,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
