"""泛雅页面只读扫描：目录任务点 + 当前内容区状态。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from autolesson.config import load_config
from autolesson.mouse import Mouse
from autolesson.ocr import OcrEngine, sort_lines
from autolesson.screen import Screen

from .catalog import FanyaCatalogScanner
from .layout import FanyaLayout
from .navigator import FanyaCatalogNavigator
from .page import FanyaPageReader

log = logging.getLogger("fanya.scan")


def run_scan(args) -> int:
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(getattr(args, "config", None), platform="fanya")
    image_path = getattr(args, "image", None)
    screen = None
    mouse = None
    if image_path:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"无法读取图片: {image_path}")
            return 2
        log.info("读取截图: %s (%dx%d)", image_path, frame.shape[1], frame.shape[0])
    else:
        screen = Screen(int(getattr(args, "monitor", 1)), log.getChild("screen"))
        mouse = Mouse(cfg, log.getChild("mouse"))
        mouse.set_origin(*screen.origin)
        # 先把正文滚到顶部，标题/播放器位置才和标定布局一致
        try:
            for _ in range(3):
                mouse.scroll_at(int(screen.width * 0.42), int(screen.height * 0.14), +10)
                time.sleep(0.2)
        except Exception:  # noqa: BLE001
            pass
        frame = screen.grab()
        log.info("截屏: %dx%d", frame.shape[1], frame.shape[0])

    ocr = OcrEngine(log.getChild("ocr"))
    full_lines = ocr.read(frame)
    full_lines = sort_lines(full_lines)
    layout = FanyaLayout.detect(frame, full_lines)
    reader = FanyaPageReader(ocr)
    page = reader.read(frame, layout)
    print(f"布局: 目录 x={layout.panel.x0}..{layout.panel.x1} y={layout.panel.y0}..{layout.panel.y1}")
    print(f"当前标题: {page.title}")
    print(f"任务标签: {list(page.tabs)} 当前: {page.active_tab}")
    print(f"任务点已完成: {page.video_completed} 时间: {page.time_text}")
    if page.current is not None:
        print(f"进度: {page.current}/{page.total} ({page.ratio:.1%})")

    scanner = FanyaCatalogScanner(ocr)
    if mouse is not None and getattr(args, "all", False):
        items = FanyaCatalogNavigator(
            scanner, screen, mouse, layout, cfg
        ).scan_all()
    else:
        items = scanner.scan(frame, layout)
    current = next((it for it in items if it.current), None)
    print(f"当前目录项: {current if current else '未识别'}")
    print(f"\n目录任务点 {len(items)} 个:")
    for item in items:
        mark = "当前" if item.current else ""
        print(f"  {item.index:<8} [{item.status:<9}] {mark:<2} {item.title}")
    if screen is not None:
        screen.close()
    return 0
