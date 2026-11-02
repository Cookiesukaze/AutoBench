"""按旧版 dover_batch 配方测量；只有显式配置源码路径才启用。"""

import os
import sys
from pathlib import Path
from ....app.remote.gpu.worker import cached


def measure(video, repo):
    import torch

    if not repo or not torch.cuda.is_available():
        raise ValueError("DOVER 需要明确的源码路径和 CUDA")
    repo = Path(repo).resolve()
    video = Path(video).resolve()
    recipe = repo / "dover.yml"
    if not recipe.is_file() or not (repo / "dover_batch.py").is_file():
        raise ValueError("DOVER 缺少 dover.yml 或 dover_batch.py")

    def build():
        previous = Path.cwd()
        try:
            os.chdir(repo)
            sys.path.insert(0, str(repo))
            from dover_batch import Evaluator

            return Evaluator("./dover.yml", "cuda")
        finally:
            os.chdir(previous)

    evaluator = cached(("dover", str(repo)), build)
    # The author's dover_batch wrapper returns [aesthetic, technical].
    aesthetic, technical = evaluator.score(str(video))
    return dict(dover_technical=float(technical), dover_aesthetic=float(aesthetic))
