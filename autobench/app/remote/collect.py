"""核对远端归档完整性后同步至本批次的测量目录。"""

from pathlib import Path
import shutil
from ..batch.inputs import sha256
from ..data.types import ContractError


def verify(folder: Path, inventory: dict):
    root = folder.resolve()
    for name, digest in inventory.items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ContractError(f"远端产物缺失或校验不符：{name}")
    return folder


def promote(source: Path, destination: Path):
    """同步本批次自有证据目录，移除上次回收留下的失效文件。"""
    source, destination = source.resolve(), destination.resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ContractError("证据来源与目标目录不能重叠")
    if not source.is_dir():
        raise ContractError(f"缺少证据目录 {source}")
    wanted = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
    if destination.exists():
        for path in destination.rglob("*"):
            if not path.resolve().is_relative_to(destination):
                raise ContractError(f"证据目标包含越界链接：{path}")
            if path.is_file() and path.relative_to(destination) not in wanted:
                path.unlink()
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination
