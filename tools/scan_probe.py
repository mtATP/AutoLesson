from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autolesson.config import load_config  # noqa: E402
from autolesson.layout import Layout  # noqa: E402
from autolesson.mouse import Mouse  # noqa: E402
from autolesson.ocr import OcrEngine  # noqa: E402
from autolesson.panel import PanelNavigator, PanelScanner  # noqa: E402
from autolesson.screen import Screen  # noqa: E402
from autolesson.platform_utils import focus_window  # noqa: E402


def main() -> int:
    cfg = load_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("probe")
    focus_window(list(cfg.window.title_keywords), log)
    screen = Screen(int(cfg.runtime.monitor), log.getChild("screen"))
    mouse = Mouse(cfg, log.getChild("mouse"))
    mouse.set_origin(*screen.origin)
    ocr = OcrEngine()
    ocr.warmup()

    frame = screen.grab()
    lines = ocr.read(frame)
    layout = Layout.detect(frame, lines, cfg)
    log.info("layout: %s", layout)
    if not layout.calibrated:
        log.error("布局未标定，无法扫描")
        return 2

    scanner = PanelScanner(ocr, cfg)
    navigator = PanelNavigator(scanner, ocr, cfg, mouse, screen, layout)
    started = time.monotonic()
    lessons = navigator.scan_all()
    elapsed = time.monotonic() - started
    log.info("扫描完成: %d 个视频，耗时 %.1fs", len(lessons), elapsed)
    for pos, lesson in enumerate(lessons, 1):
        log.info("%03d %s", pos, lesson)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
