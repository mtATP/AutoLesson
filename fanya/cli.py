"""泛雅版命令行入口。

当前先提供只读的页面探针：把泛雅课程页截图 OCR 成带坐标的文本，
用于确定目录、播放器、任务点和弹窗的位置，再写平台专用识别。
"""

from __future__ import annotations

import argparse
import logging
import sys

from .probe import run_probe
from .run import run_watcher
from .scan import run_scan


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="配置文件；默认 config.yaml")
    parser.add_argument("--monitor", type=int, default=1, help="多显示器编号，从 1 开始")
    parser.add_argument("--image", help="截图路径：只做规划，不点击")
    parser.add_argument("--dry-run", action="store_true", help="只扫描目录，不点击")
    parser.add_argument("--max-items", type=int, default=0, help="最多处理几个任务点")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fanya",
        description="AutoLesson 泛雅版：纯视觉课程挂机（开发中）",
    )
    sub = parser.add_subparsers(dest="command")

    probe = sub.add_parser(
        "probe",
        help="OCR 当前屏幕或指定截图，输出文字坐标并生成标注图（只读）",
    )
    probe.add_argument("--image", help="截图路径；不填则截取当前屏幕")
    probe.add_argument("--monitor", type=int, default=1, help="多显示器编号，从 1 开始")
    probe.add_argument(
        "--crop",
        help="只分析区域，格式 x0,y0,x1,y1（像素，相对整张图）",
    )
    probe.add_argument("--out", help="标注图输出路径；默认 runtime/fanya-probe-时间.png")
    probe.add_argument("--verbose", action="store_true")

    scan = sub.add_parser(
        "scan",
        help="只读扫描泛雅课程页：目录任务点、当前标题、视频进度",
    )
    scan.add_argument("--image", help="截图路径；不填则截取当前屏幕")
    scan.add_argument("--config", help="配置文件；默认 config.yaml")
    scan.add_argument("--monitor", type=int, default=1, help="多显示器编号，从 1 开始")
    scan.add_argument("--all", action="store_true", help="滚动扫描整个目录（需要实时屏幕）")
    scan.add_argument("--verbose", action="store_true")

    run = sub.add_parser("run", help="泛雅视频挂机（只处理视频任务点）")
    _add_run_args(run)
    run.set_defaults(mode="video")
    practice = sub.add_parser("practice", help="泛雅练习模式：只暂时保存，绝不提交")
    _add_run_args(practice)
    practice.set_defaults(mode="practice")
    auto = sub.add_parser("auto", help="泛雅列表模式：按目录顺序处理视频和练习")
    _add_run_args(auto)
    auto.set_defaults(mode="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "probe":
        return run_probe(args)
    if args.command == "scan":
        return run_scan(args)
    if args.command in ("run", "practice", "auto"):
        return run_watcher(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
