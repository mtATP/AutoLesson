"""只读页面探针：不点击、不滚动，只做截图 + OCR + 坐标输出。

用法：
    python -m fanya probe
    python -m fanya probe --image 泛雅课程页.png
    python -m fanya probe --crop 0,0,500,600

输出每行：矩形(x0,y0,x1,y1) 置信度 文本；同时生成带检测框的标注图。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from autolesson.ocr import OcrEngine, sort_lines
from autolesson.screen import Screen
from autolesson.types import OcrLine, Rect

log = logging.getLogger("fanya.probe")


def parse_crop(value: str | None) -> Rect | None:
    if not value:
        return None
    parts = [int(x.strip()) for x in value.replace("，", ",").split(",")]
    if len(parts) != 4:
        raise ValueError("--crop 需要 4 个整数：x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--crop 的 x1/y1 必须大于 x0/y0")
    return Rect(x0, y0, x1, y1)


def annotate(image: np.ndarray, lines: list[OcrLine]) -> np.ndarray:
    canvas = image.copy()
    for line in lines:
        rect = line.rect
        cv2.rectangle(canvas, (rect.x0, rect.y0), (rect.x1, rect.y1), (0, 200, 0), 1)
        label = f"{line.score:.2f}"
        cv2.putText(
            canvas,
            label,
            (rect.x0, max(12, rect.y0 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _default_out() -> Path:
    return Path("runtime") / time.strftime("fanya-probe-%Y%m%d-%H%M%S.png")


def run_probe(args) -> int:
    # Windows 控制台默认 GBK，OCR 文本含生僻符号时会直接抛 UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    crop = None
    try:
        crop = parse_crop(getattr(args, "crop", None))
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2

    image_path = getattr(args, "image", None)
    if image_path:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}", file=sys.stderr)
            return 2
        log.info("读取截图: %s (%dx%d)", image_path, image.shape[1], image.shape[0])
    else:
        screen = Screen(int(getattr(args, "monitor", 1)), log.getChild("screen"))
        try:
            image = screen.grab()
            log.info("已截取屏幕 (%dx%d)", image.shape[1], image.shape[0])
        finally:
            screen.close()

    if crop is not None:
        h, w = image.shape[:2]
        box = Rect(
            max(0, crop.x0), max(0, crop.y0), min(w, crop.x1), min(h, crop.y1)
        )
        image = image[box.y0 : box.y1, box.x0 : box.x1]
        log.info("分析区域: %s", box)

    ocr = OcrEngine(log.getChild("ocr"))
    lines = ocr.read(image)
    lines = sort_lines(lines, row_tolerance=max(6, int(0.012 * image.shape[1])))
    print(f"检测到 {len(lines)} 行：")
    for line in lines:
        rect = line.rect
        print(f"({rect.x0},{rect.y0},{rect.x1},{rect.y1})\t{line.score:.2f}\t{line.text}")

    out = Path(getattr(args, "out", None) or _default_out())
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), annotate(image, lines)):
        print(f"标注图保存失败: {out}", file=sys.stderr)
        return 2
    print(f"标注图: {out.resolve()}")
    return 0
