from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from . import __version__, vision
from .config import ConfigError, load_config
from .controller import AutoLesson
from .hotkey import HotkeyListener
from .layout import Layout
from .llm import LLMClient
from .mouse import InputUnavailable, Mouse
from .notify import Notifier
from .ocr import OcrEngine, OcrUnavailable
from .panel import PanelNavigator, PanelScanner
from .practice import PracticePage, PracticeRunner
from .player import PlayerMonitor
from .quiz import parse_quiz
from .screen import Screen, ScreenUnavailable
from .selftest import run_selftest
from .state import State
from .types import FatalError, Rect, UserAbort


def setup_logging(cfg, verbose: bool = False) -> logging.Logger:
    log_dir = cfg.path(str(cfg.runtime.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / time.strftime("autolesson-%Y%m%d-%H%M%S.log")
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    root.addHandler(file_handler)
    root.addHandler(console)

    for noisy in ("RapidOCR", "onnxruntime", "PIL", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    log = logging.getLogger("autolesson")
    log.info("日志文件: %s", log_file)
    return log


def configure_console() -> None:
    # 保留控制台原有编码（Windows CMD 可能是 GBK），只把不可编码字符替换掉
    try:
        if sys.stdout.encoding:
            sys.stdout.reconfigure(errors="replace")
        if sys.stderr.encoding:
            sys.stderr.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _check_dependencies(dry_run: bool) -> None:
    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")
    try:
        import mss  # noqa: F401
    except ImportError:
        missing.append("mss")
    try:
        import rapidocr  # noqa: F401
    except ImportError:
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except ImportError:
            missing.append("rapidocr onnxruntime")
    if not dry_run:
        try:
            import pyautogui  # noqa: F401
        except Exception:  # noqa: BLE001
            missing.append("pyautogui")
    if missing:
        raise RuntimeError(
            "缺少依赖: " + ", ".join(missing) + "\n请执行: pip install -r requirements.txt"
        )


def _make_ocr(cfg, logger):
    return OcrEngine(
        logger,
        intra_threads=int(cfg.ocr.intra_threads),
        inter_threads=int(cfg.ocr.inter_threads),
        cv_threads=int(cfg.ocr.cv_threads),
        det_limit_side=int(cfg.ocr.det_limit_side),
        use_cls=bool(cfg.ocr.use_cls),
    )


def _common_overrides(args) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if getattr(args, "dry_run", False):
        overrides.setdefault("runtime", {})["dry_run"] = True
    if getattr(args, "max_lessons", None) is not None:
        overrides.setdefault("runtime", {})["max_lessons"] = int(args.max_lessons)
    if getattr(args, "max_minutes", None) is not None:
        overrides.setdefault("runtime", {})["max_minutes"] = int(args.max_minutes)
    if getattr(args, "monitor", None) is not None:
        overrides.setdefault("runtime", {})["monitor"] = int(args.monitor)
    return overrides


def _load_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"截图不存在: {p}")
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        image = cv2.imread(str(p))
    if image is None:
        raise ValueError(f"无法解码图片: {p}")
    return image


def _annotate_layout(
    image: np.ndarray,
    layout: Layout,
    lessons,
    state,
    cards,
    out_path: Path,
) -> None:
    canvas = image.copy()
    cv2.rectangle(
        canvas, (layout.panel_left, layout.panel_top), (layout.panel_right, layout.panel_bottom), (0, 165, 255), 2
    )
    cv2.putText(canvas, "course panel", (layout.panel_left + 4, max(15, layout.panel_top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    colors = {
        "completed": (0, 160, 0),
        "in_progress": (0, 120, 255),
        "not_started": (90, 90, 90),
    }
    for lesson in lessons:
        color = colors.get(lesson.status, (0, 0, 0))
        cv2.rectangle(canvas, (lesson.title_rect.x0, lesson.title_rect.y0), (lesson.title_rect.x1, lesson.title_rect.y1), color, 2)
        label = f"{lesson.index or '?'} {lesson.status}"
        cv2.putText(canvas, label, (lesson.title_rect.x0, max(12, lesson.title_rect.y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    if state is not None:
        cv2.line(canvas, (0, state.bar_y), (layout.panel_left, state.bar_y), (0, 255, 0), 2)
        if state.play_button is not None:
            cv2.rectangle(
                canvas,
                (state.play_button.x0, state.play_button.y0),
                (state.play_button.x1, state.play_button.y1),
                (255, 0, 255),
                2,
            )
    for card in cards:
        cv2.rectangle(canvas, (card.x0, card.y0), (card.x1, card.y1), (255, 0, 0), 2)
    Screen.save(out_path, canvas)


def _require_llm_for_mode(cfg, log) -> bool:
    if cfg.llm_ready():
        return True
    log.error(
        "当前是无 LLM 模式（llm.enabled=false 或未配置 API Key）："
        "只允许 run 视频模式，auto/practice 已禁用。"
    )
    return False


def cmd_run(args) -> int:
    cfg = load_config(args.config, _common_overrides(args))
    log = setup_logging(cfg, args.verbose)
    if bool(getattr(args, "list_mode", False)) and not _require_llm_for_mode(cfg, log):
        return 2
    _check_dependencies(bool(cfg.runtime.dry_run))
    if cfg.runtime.dry_run:
        log.warning("dry-run 模式：只分析屏幕，不移动鼠标、不点击")

    if bool(cfg.window.focus):
        from .platform_utils import focus_window

        focus_window(list(cfg.window.title_keywords), log)

    state = State(cfg.path(str(cfg.runtime.state_file)), log.getChild("state"))
    if getattr(args, "clear_record", False):
        state.clear_records()
    notifier = Notifier(cfg, log.getChild("notify"))
    ocr = _make_ocr(cfg, log.getChild("ocr"))
    ocr.warmup()
    try:
        screen = Screen(int(cfg.runtime.monitor), log.getChild("screen"))
        mouse = Mouse(cfg, log.getChild("mouse"), dry_run=bool(cfg.runtime.dry_run))
        mouse.set_origin(*screen.origin)
    except (ScreenUnavailable, InputUnavailable) as exc:
        notifier.send("AutoLesson 无法启动", str(exc), "fatal", force=True)
        raise

    stop_event = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def on_sigint(_signum, _frame):
        if stop_event.is_set():
            raise KeyboardInterrupt
        stop_event.set()
        log.warning("收到 Ctrl+C，正在安全停止（再按一次强制退出）...")

    try:
        signal.signal(signal.SIGINT, on_sigint)
    except ValueError:
        pass

    controller = AutoLesson(
        cfg,
        ocr,
        screen,
        mouse,
        state,
        notifier,
        log.getChild("controller"),
        stop_event,
        list_mode=bool(getattr(args, "list_mode", False)),
    )
    hotkey = HotkeyListener(
        str(cfg.runtime.stop_hotkey),
        stop_event,
        log.getChild("hotkey"),
        extra_keys=list(cfg.get("runtime.extra_stop_hotkeys", ["ESC"]) or []),
    )
    hotkey.start()
    try:
        return controller.run()
    finally:
        hotkey.stop()
        try:
            signal.signal(signal.SIGINT, previous_handler)
        except (ValueError, TypeError):
            pass


def cmd_practice(args) -> int:
    cfg = load_config(args.config, _common_overrides(args))
    log = setup_logging(cfg, args.verbose)
    if not _require_llm_for_mode(cfg, log):
        return 2
    _check_dependencies(bool(cfg.runtime.dry_run))
    if not args.page and bool(cfg.window.focus):
        from .platform_utils import focus_window

        focus_window(list(cfg.window.title_keywords), log)

    state = State(cfg.path(str(cfg.runtime.state_file)), log.getChild("state"))
    if getattr(args, "clear_record", False):
        state.clear_records()
    notifier = Notifier(cfg, log.getChild("notify"))
    ocr = _make_ocr(cfg, log.getChild("ocr"))
    ocr.warmup()
    try:
        screen = Screen(int(cfg.runtime.monitor), log.getChild("screen"))
        mouse = Mouse(cfg, log.getChild("mouse"), dry_run=bool(cfg.runtime.dry_run))
        mouse.set_origin(*screen.origin)
    except (ScreenUnavailable, InputUnavailable) as exc:
        notifier.send("AutoLesson 自动练习无法启动", str(exc), "fatal", force=True)
        raise

    stop_event = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def on_sigint(_signum, _frame):
        if stop_event.is_set():
            raise KeyboardInterrupt
        stop_event.set()
        log.warning("收到 Ctrl+C，正在安全停止练习（再按一次强制退出）...")

    try:
        signal.signal(signal.SIGINT, on_sigint)
    except ValueError:
        pass

    hotkey = HotkeyListener(
        str(cfg.runtime.stop_hotkey),
        stop_event,
        log.getChild("hotkey"),
        extra_keys=list(cfg.get("runtime.extra_stop_hotkeys", ["ESC"]) or []),
    )
    hotkey.start()
    llm = LLMClient(cfg, state, log.getChild("llm"))
    try:
        if args.page:
            page = PracticePage(
                ocr, cfg, mouse, screen, llm, state, log.getChild("practice"), stop_event,
                notifier=notifier,
            )
            result = page.run(close_tab=not args.keep_open)
            log.info("当前练习页处理结果: %s", result)
            if result.get("not_ready"):
                notifier.send(
                    "自动练习页面未就绪",
                    "练习页在超时时间内仍是白屏/未加载完整；脚本未把它当作完成，可稍后重试。",
                    "warning",
                )
            elif result.get("opened"):
                notifier.send(
                    "自动练习完成",
                    f"已答 {result.get('answered', 0)} 题，跳过已答 {result.get('skipped', 0)} 题，"
                    f"暂存={'成功' if result.get('saved') else '未确认'}",
                    "info",
                )
            return 0 if result.get("opened") else 2

        frame = screen.grab()
        lines = ocr.read(frame)
        layout = Layout.detect(frame, lines, cfg)
        if not layout.calibrated:
            time.sleep(1.0)
            frame = screen.grab()
            lines = ocr.read(frame)
            refreshed = Layout.detect(frame, lines, cfg)
            if refreshed.calibrated:
                layout = refreshed
        scanner = PanelScanner(ocr, cfg)
        navigator = PanelNavigator(scanner, ocr, cfg, mouse, screen, layout)
        runner = PracticeRunner(
            ocr, cfg, mouse, screen, llm, state, navigator,
            log.getChild("practice"), stop_event, notifier,
        )
        max_tests = int(args.max_tests if args.max_tests is not None else cfg.practice.max_tests)
        result = runner.run(max_tests=max_tests)
        log.info("自动练习汇总: %s", result)
        notifier.send(
            "AutoLesson 自动练习完成",
            f"处理 {result.get('tests', 0)} 个平时测试，答 {result.get('answered', 0)} 题，"
            f"跳过已答 {result.get('skipped', 0)} 题，跳过已提交页面 {result.get('submitted', 0)} 个，"
            f"点击前跳过已完成 {result.get('completed_skipped', 0)} 个。\n"
            "脚本只执行了暂存，未提交任何作业。",
            "info",
        )
        return 0
    except UserAbort as exc:
        log.warning("练习中止: %s", exc)
        state.save()
        notifier.send("自动练习已中止", str(exc), "warning", force=True)
        return 130
    except FatalError as exc:
        log.error("练习致命故障: %s | %s", exc.reason, exc.detail)
        state.save()
        notifier.send("自动练习已停止", f"{exc.reason}\n{exc.detail}", "fatal", force=True)
        return 2
    finally:
        hotkey.stop()
        state.save()
        try:
            signal.signal(signal.SIGINT, previous_handler)
        except (ValueError, TypeError):
            pass


def cmd_calibrate(args) -> int:
    cfg = load_config(args.config, _common_overrides(args))
    log = setup_logging(cfg, args.verbose)
    _check_dependencies(True)
    ocr = _make_ocr(cfg, log.getChild("ocr"))
    ocr.warmup()

    if args.image:
        frame = _load_image(args.image)
        screen = None
    else:
        from .platform_utils import focus_window

        if bool(cfg.window.focus):
            focus_window(list(cfg.window.title_keywords), log)
        screen = Screen(int(cfg.runtime.monitor), log.getChild("screen"))
        time.sleep(1.0)
        frame = screen.grab()

    lines = ocr.read(frame)
    layout = Layout.detect(frame, lines, cfg)
    scanner = PanelScanner(ocr, cfg)
    lessons = scanner.scan(frame, layout)
    player = PlayerMonitor(ocr, cfg)
    state = player.snapshot(frame, layout)
    cards = vision.find_cards(
        frame, layout.panel_left, int(cfg.overlay.card_threshold), int(cfg.overlay.card_close_kernel)
    )

    print("\n" + "=" * 64)
    print(f"屏幕尺寸: {layout.screen_w}x{layout.screen_h}")
    print(f"目录面板: x={layout.panel_left}..{layout.panel_right} y={layout.panel_top}..{layout.panel_bottom}")
    print(f"播放器宽: {layout.player_width}")
    print(f"自动识别: {'是' if layout.calibrated else '否（使用比例回退）'}")
    print("\n建议写入 config.yaml:")
    print("layout:")
    print(f"  panel_left: {layout.panel_left}")
    print(f"  panel_top: {layout.panel_top}")
    print("\n目录视频:")
    for lesson in lessons:
        print(f"  {lesson}")
    if not lessons:
        print("  (未识别到视频，请确认已打开课程学习页并保持窗口可见)")
    if state is not None:
        print(f"\n播放器: {state.raw_time} 播放={state.playing} 进度={state.percent:.1f}%")
    else:
        print("\n播放器: 未检测到进度条（控件可能隐藏，或尚未打开视频）")
    if cards:
        print("\n检测到的浅色弹窗:")
        for card in cards:
            header = ocr.read_text(
                Rect(card.x0, card.y0, card.x1, card.y0 + max(90, int(0.28 * card.height))).crop(frame),
                offset=(card.x0, card.y0),
                upscale=1.3,
            )
            kind = AutoLesson._classify_overlay(header)
            print(f"  {card} -> {kind}: {header[:60]!r}")
    out = Path(args.out) if args.out else cfg.path(str(cfg.runtime.shot_dir)) / "calibration.png"
    _annotate_layout(frame, layout, lessons, state, cards, out)
    print(f"\n标注图已保存: {out}")
    print("=" * 64 + "\n")
    if screen is not None:
        screen.close()
    return 0


def cmd_inspect(args) -> int:
    cfg = load_config(args.config, _common_overrides(args))
    setup_logging(cfg, args.verbose)
    _check_dependencies(True)
    ocr = _make_ocr(cfg, logging.getLogger("autolesson.ocr"))
    ocr.warmup()
    frame = _load_image(args.image)
    layout = Layout.detect(frame, ocr.read(frame), cfg)
    scanner = PanelScanner(ocr, cfg)
    lessons = scanner.scan(frame, layout)
    player = PlayerMonitor(ocr, cfg)
    state = player.snapshot(frame, layout)
    cards = vision.find_cards(
        frame, layout.panel_left, int(cfg.overlay.card_threshold), int(cfg.overlay.card_close_kernel)
    )

    print("=" * 72)
    print(f"图片: {args.image}  {frame.shape[1]}x{frame.shape[0]}")
    print(f"布局: panel_left={layout.panel_left} panel_top={layout.panel_top} 自动识别={layout.calibrated}")
    print(f"视频数: {len(lessons)}")
    for lesson in lessons:
        print(f"  {lesson}  title_rect={lesson.title_rect} row={lesson.row}")
    if state is not None:
        print(
            f"播放器: current={state.current} total={state.total} ratio={state.ratio:.4f} "
            f"playing={state.playing} bar_y={state.bar_y} raw={state.raw_time!r}"
        )
    else:
        print("播放器: 无进度条")
    for index, card in enumerate(cards):
        header = ocr.read_text(
            Rect(card.x0, card.y0, card.x1, card.y0 + max(90, int(0.28 * card.height))).crop(frame),
            offset=(card.x0, card.y0),
            upscale=1.3,
        )
        footer = ocr.read_text(
            Rect(card.x0, max(card.y0, card.y1 - max(60, int(0.16 * card.height))), card.x1, card.y1).crop(frame),
            offset=(card.x0, card.y0),
            upscale=1.3,
        )
        kind = AutoLesson._classify_overlay(header + "\n" + footer)
        print(f"\n弹窗 {index}: {card} -> {kind}")
        print(f"  header: {header[:120]!r}")
        print(f"  footer: {footer[:120]!r}")
        if kind == "quiz":
            lines = ocr.read(card.crop(frame), offset=(card.x0, card.y0), upscale=1.6)
            question = parse_quiz(lines, card, frame)
            print(f"  题型: {question.kind} 需要视觉: {question.needs_vision}")
            print(f"  题干: {question.text}")
            for letter, text in question.options.items():
                print(f"    {letter}. {text}  {question.option_rects.get(letter)}")
    if args.annotate:
        _annotate_layout(frame, layout, lessons, state, cards, Path(args.annotate))
        print(f"\n标注图: {args.annotate}")
    print("=" * 72)
    return 0


def cmd_selftest(args) -> int:
    return run_selftest(Path(args.fixtures), Path(args.out))


def cmd_gui(args) -> int:
    from .gui import main as gui_main

    return gui_main(args.config)


def cmd_version(_args) -> int:
    print(f"AutoLesson {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autolesson",
        description="基于纯视觉的智慧树网课自动观看助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m autolesson calibrate\n"
            "  python -m autolesson inspect --image runtime/shots/debug.png\n"
            "  python -m autolesson gui\n"
            "  python -m autolesson selftest\n"
            "  python -m autolesson run\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"AutoLesson {__version__}")
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default="config.yaml", help="配置文件路径（默认 config.yaml）")
    common.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")

    run = sub.add_parser("run", parents=[common], help="开始挂机观看")
    run.add_argument("--dry-run", action="store_true", help="只分析不下手，用于验证识别")
    run.add_argument("--max-lessons", type=int, default=None, help="本次最多完成几个视频")
    run.add_argument("--max-minutes", type=int, default=None, help="本次最多运行多少分钟")
    run.add_argument("--monitor", type=int, default=None, help="显示器编号，从 1 开始")
    run.add_argument("--clear-record", action="store_true", help="运行前清空已完成记录（视频/练习），保留答题缓存")
    run.set_defaults(func=cmd_run, list_mode=False)

    auto = sub.add_parser(
        "auto",
        parents=[common],
        help="按目录列表顺序自动完成视频和平时测试（视频/题目谁在上面先做谁）",
    )
    auto.add_argument("--dry-run", action="store_true", help="只分析不下手")
    auto.add_argument("--max-lessons", type=int, default=None, help="本次最多完成几个视频")
    auto.add_argument("--max-minutes", type=int, default=None, help="本次最多运行多少分钟")
    auto.add_argument("--monitor", type=int, default=None, help="显示器编号，从 1 开始")
    auto.add_argument("--clear-record", action="store_true", help="运行前清空已完成记录（视频/练习），保留答题缓存")
    auto.set_defaults(func=cmd_run, list_mode=True)

    practice = sub.add_parser(
        "practice", parents=[common], help="自动完成平时测试（截图 LLM，只暂存，绝不提交）"
    )
    practice.add_argument("--page", action="store_true", help="只处理当前已打开的练习标签页")
    practice.add_argument("--keep-open", action="store_true", help="--page 处理后不关闭练习标签页")
    practice.add_argument("--max-tests", type=int, default=None, help="本次最多处理几个平时测试")
    practice.add_argument("--dry-run", action="store_true", help="只分析不点击")
    practice.add_argument("--monitor", type=int, default=None, help="显示器编号，从 1 开始")
    practice.add_argument("--clear-record", action="store_true", help="运行前清空已完成记录（视频/练习），保留答题缓存")
    practice.set_defaults(func=cmd_practice)

    cal = sub.add_parser("calibrate", parents=[common], help="截屏并标定面板/播放器位置")
    cal.add_argument("--image", help="使用已有截图而不是实时截屏")
    cal.add_argument("--out", help="标注图输出路径")
    cal.add_argument("--monitor", type=int, default=None, help="显示器编号")
    cal.set_defaults(func=cmd_calibrate)

    ins = sub.add_parser("inspect", parents=[common], help="离线分析一张截图")
    ins.add_argument("--image", required=True, help="截图路径")
    ins.add_argument("--annotate", help="输出标注图路径")
    ins.set_defaults(func=cmd_inspect)

    gui = sub.add_parser(
        "gui", parents=[common], help="打开图形控制台（选择模式、配置、启动/停止）"
    )
    gui.set_defaults(func=cmd_gui)

    test = sub.add_parser("selftest", help="用内置截图做离线自检")
    test.add_argument("--fixtures", default="tests/fixtures", help="截图目录")
    test.add_argument("--out", default="runtime/selftest", help="标注输出目录")
    test.set_defaults(func=cmd_selftest)

    version = sub.add_parser("version", help="显示版本")
    version.set_defaults(func=cmd_version)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1
    except (OcrUnavailable, ScreenUnavailable, InputUnavailable) as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"文件错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("autolesson").exception("命令执行失败")
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
