"""泛雅课程页布局识别。

页面结构（mooc1.chaoxing.com/mycourse/studentstudy，mooc2=1）：
- 顶部：返回课程 / 章节详情；
- 左侧：章节标题、任务标签、视频区；
- 右侧：目录/讨论/笔记/标注侧栏，目录列表在 #coursetree 内滚动。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from autolesson.types import OcrLine, Rect

log = logging.getLogger("fanya.layout")


@dataclass
class FanyaLayout:
    screen_w: int
    screen_h: int
    panel: Rect
    list_rect: Rect
    content: Rect
    calibrated: bool = False

    @property
    def panel_width(self) -> int:
        return self.panel.width

    def to_dict(self) -> dict:
        return {
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "panel": self.panel.as_dict(),
            "list": self.list_rect.as_dict(),
            "content": self.content.as_dict(),
            "calibrated": self.calibrated,
        }

    @classmethod
    def detect(
        cls,
        frame: np.ndarray,
        lines: list[OcrLine],
        panel_left_ratio: float = 0.74,
        bottom_margin: int = 40,
    ) -> "FanyaLayout":
        h, w = frame.shape[:2]
        label: Optional[OcrLine] = None
        for line in lines:
            text = line.text.replace(" ", "")
            if "目录" in text and line.rect.cx > 0.60 * w and line.rect.y0 < 0.30 * h:
                if label is None or line.rect.x0 > label.rect.x0:
                    label = line
        if label is not None:
            panel_left = int(label.rect.x0 - 0.02 * w)
            calibrated = True
        else:
            panel_left = int(panel_left_ratio * w)
            calibrated = False
        panel_left = int(np.clip(panel_left, int(0.60 * w), int(0.90 * w)))
        # 目录标签在侧栏顶部；列表从标签栏和搜索框下方开始
        panel_top = int(0.095 * h)
        if label is not None:
            panel_top = max(int(0.05 * h), int(label.rect.y0 - 0.01 * h))
        list_top = panel_top + int(0.085 * h)
        bottom = max(panel_top + 200, h - max(0, bottom_margin))
        panel = Rect(panel_left, panel_top, w, bottom)
        list_rect = Rect(panel_left, list_top, w, bottom)
        content = Rect(0, 0, panel_left, bottom)
        layout = cls(
            screen_w=w,
            screen_h=h,
            panel=panel,
            list_rect=list_rect,
            content=content,
            calibrated=calibrated,
        )
        log.info(
            "泛雅布局: 屏幕 %dx%d, 目录 x=%d..%d y=%d..%d, 内容 x=0..%d",
            w,
            h,
            panel.x0,
            panel.x1,
            panel.y0,
            panel.y1,
            panel_left,
        )
        return layout
