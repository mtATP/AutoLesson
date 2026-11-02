"""AutoLesson 统一入口。

- ``python -m autolesson`` / ``python -m autolesson gui``：统一 GUI；
- ``python -m autolesson zhihuishu ...``：智慧树版 CLI；
- ``python -m autolesson fanya ...``：泛雅版 CLI；
- 直接 ``python -m autolesson run`` 等仍按智慧树版兼容。
"""

from __future__ import annotations

import sys
from typing import Optional


def _gui_config(argv: list[str]) -> str:
    try:
        index = argv.index("--config")
        return argv[index + 1]
    except (ValueError, IndexError):
        return "config.yaml"


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "gui":
        from .gui import main as gui_main

        return gui_main(_gui_config(args))
    if args[0] == "zhihuishu":
        from zhihuishu.cli import main as zhs_main

        return zhs_main(args[1:])
    if args[0] == "fanya":
        from fanya.cli import main as fanya_main

        return fanya_main(args[1:])
    # 兼容旧命令：python -m autolesson run/auto/practice/...
    from zhihuishu.cli import main as zhs_main

    return zhs_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
