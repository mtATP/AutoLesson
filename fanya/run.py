"""泛雅视频挂机命令行入口（第一版）。"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2

from autolesson.config import load_config
from autolesson.llm import LLMClient
from autolesson.mouse import Mouse
from autolesson.ocr import OcrEngine
from autolesson.screen import Screen
from autolesson.hotkey import HotkeyListener
from autolesson.state import State
from autolesson.types import UserAbort

from .controller import FanyaWatcher


def setup_fanya_logging(cfg) -> Path:
    """控制台之外再写一份持久日志；视频完成记录不依赖目录绿勾。"""
    log_dir = cfg.path(str(cfg.runtime.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / time.strftime("fanya-%Y%m%d-%H%M%S.log")
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "").endswith(
            path.name
        ):
            return path
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    return path


class StaticScreen:
    def __init__(self, image):
        self.image = image

    def grab(self):
        return self.image

    def close(self):
        pass


class DryMouse:
    def click_rect(self, rect, label=""):
        pass

    def click(self, x, y, label=""):
        pass

    def scroll_at(self, x, y, clicks):
        pass

    def move_to(self, x, y, duration=0.2):
        pass

    def set_origin(self, *args):
        pass


def run_watcher(args) -> int:
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(getattr(args, "config", None), platform="fanya")
    log_file = setup_fanya_logging(cfg)
    log = logging.getLogger("fanya.run")
    log.info("泛雅运行日志: %s", log_file)
    state = State(cfg.path("runtime/fanya_state.json"), logging.getLogger("fanya.state"))
    state.note_run()
    mode = str(getattr(args, "mode", "video") or "video")
    llm = LLMClient(cfg, state, logging.getLogger("fanya.llm")) if cfg.quiz.enabled else None
    if mode in ("practice", "auto") and llm is None:
        logging.getLogger("fanya.run").error("练习/列表模式需要配置大模型（quiz.enabled 且有 API Key）")
        return 2
    ocr = OcrEngine(logging.getLogger("fanya.ocr"))
    ocr.warmup()

    image_path = getattr(args, "image", None)
    if image_path:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            return 2
        screen = StaticScreen(image)
        mouse = DryMouse()
        dry_run = True
        logging.getLogger("fanya.run").info("图片模式：只做规划，不点击")
    else:
        screen = Screen(int(getattr(args, "monitor", 1)), logging.getLogger("fanya.screen"))
        mouse = Mouse(cfg, logging.getLogger("fanya.mouse"))
        mouse.set_origin(*screen.origin)
        dry_run = bool(getattr(args, "dry_run", False))

    stop_event = threading.Event()
    hotkey = HotkeyListener(
        str(cfg.runtime.stop_hotkey),
        stop_event,
        logging.getLogger("fanya.hotkey"),
        extra_keys=list(cfg.get("runtime.extra_stop_hotkeys", ["ESC"]) or []),
    )
    hotkey.start()
    watcher = FanyaWatcher(
        cfg,
        ocr,
        screen,
        mouse,
        logging.getLogger("fanya.controller"),
        dry_run=dry_run,
        state=state,
        llm=llm,
        stop_event=stop_event,
        mode=mode,
    )
    try:
        if dry_run:
            pending = watcher.plan()
            logging.getLogger("fanya.run").info(
                "dry-run 待处理 %d 个任务点:", len(pending)
            )
            for item in pending[: int(getattr(args, "max_items", 0) or 0) or None]:
                logging.getLogger("fanya.run").info(
                    "  %s %s [%s]", item.index, item.title, item.status
                )
            return 0
        return watcher.run(max_items=int(getattr(args, "max_items", 0) or 0))
    except UserAbort as exc:
        logging.getLogger("fanya.run").warning("用户中止: %s", exc)
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("fanya.run").exception("运行失败: %s", exc)
        return 2
    finally:
        hotkey.stop()
        state.save()
        try:
            screen.close()
        except Exception:  # noqa: BLE001
            pass
