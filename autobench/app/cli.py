"""命令行入口：plan / quality / judge / score。"""

import sys

COMMANDS = ("plan", "quality", "judge", "score")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit("usage: autobench {%s} ..." % "|".join(COMMANDS))
    raise NotImplementedError(f"{argv[0]} 尚未实现")
