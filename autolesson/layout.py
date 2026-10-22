from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import vision
from .config import Config
from .types import OcrLine, Rect

log = logging.getLogger("autolesson.layout")


@dataclass
class Layout:
    screen_w: int
    screen_h: int
    panel_left: int
    panel_right: int
    panel_top: int
    panel_bottom: int
    calibrated: bool = False

    @property
    def panel(self) -> Rect:
        return Rect(self.panel_left, self.panel_top, self.panel_right, self.panel_bottom)

    @property
    def player_width(self) -> int:
        return max(1, self.panel_left)

    @property
    def player_center(self) -> tuple[int, int]:
        # 真正居中；旧值 0.55/0.55 会让点击偏右下，容易落到控制条/下一节附近
        return int(self.panel_left * 0.50), int(self.screen_h * 0.50)

    @classmethod
    def detect(
        cls,
        frame: np.ndarray,
        lines: list[OcrLine],
        cfg: Config,
    ) -> "Layout":
        h, w = frame.shape[:2]
        label = vision.find_directory_label(lines, w) if bool(cfg.layout.auto_calibrate) else None
        panel_left = int(round(w * float(cfg.layout.panel_left_ratio)))
        panel_top = int(round(h * float(cfg.layout.panel_top_ratio)))
        calibrated = False
        if label is not None:
            panel_left = int(label.x0 - 0.012 * w)
            panel_top = int(label.y1 + 0.012 * h)
            calibrated = True
        if int(cfg.layout.panel_left) > 0:
            panel_left = int(cfg.layout.panel_left)
            calibrated = True
        if int(cfg.layout.panel_top) > 0:
            panel_top = int(cfg.layout.panel_top)
            calibrated = True
        panel_left = int(np.clip(panel_left, int(0.45 * w), int(0.95 * w)))
        panel_top = int(np.clip(panel_top, int(0.05 * h), int(0.55 * h)))
        margin = int(cfg.layout.panel_right_margin)
        panel_right = max(panel_left + 120, w - margin)
        layout = cls(
            screen_w=w,
            screen_h=h,
            panel_left=panel_left,
            panel_right=panel_right,
            panel_top=panel_top,
            panel_bottom=h,
            calibrated=calibrated,
        )
        log.info(
            "布局: 屏幕 %dx%d, 播放器宽 %d, 目录区域 x=%d..%d y=%d..%d (%s)",
            w,
            h,
            layout.player_width,
            layout.panel_left,
            layout.panel_right,
            layout.panel_top,
            layout.panel_bottom,
            "自动识别" if calibrated else "比例回退",
        )
        return layout

    @classmethod
    def manual(
        cls,
        w: int,
        h: int,
        panel_left: int,
        panel_right: Optional[int] = None,
        panel_top: int = 0,
    ) -> "Layout":
        return cls(
            screen_w=w,
            screen_h=h,
            panel_left=panel_left,
            panel_right=panel_right if panel_right is not None else w,
            panel_top=panel_top,
            panel_bottom=h,
            calibrated=True,
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "panel_left": self.panel_left,
            "panel_right": self.panel_right,
            "panel_top": self.panel_top,
            "panel_bottom": self.panel_bottom,
            "calibrated": self.calibrated,
        }
