"""泛雅课程内容区（标题/任务标签/视频状态/时间）识别。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from autolesson.ocr import OcrEngine, sort_lines
from autolesson.types import OcrLine, Rect

from .layout import FanyaLayout

log = logging.getLogger("fanya.page")

TIME_RANGE_RE = re.compile(
    r"(\d{1,3})\s*[:：]\s*(\d{1,2})\s*[/／]\s*(\d{1,3})\s*[:：]\s*(\d{1,2})"
)
COMPLETED_WORDS = ("任务点已完成", "已完成")
UI_TITLE_WORDS = {
    "返回课程",
    "章节详情",
    "目录",
    "讨论",
    "笔记",
    "标注",
    "搜索",
    "视频",
    "章节测验",
    "任务点已完成",
    "知识点",
    "写笔记",
    "AI助教",
}


@dataclass
class FanyaPageInfo:
    title: str
    tabs: dict[str, Rect] = field(default_factory=dict)
    active_tab: str = ""
    video_completed: bool = False
    time_text: str = ""
    current: Optional[int] = None
    total: Optional[int] = None
    video_rect: Rect = Rect(0, 0, 0, 0)
    play_button: Optional[Rect] = None

    @property
    def ratio(self) -> Optional[float]:
        if self.current is None or not self.total:
            return None
        return max(0.0, min(1.0, self.current / self.total))


class FanyaPageReader:
    def __init__(self, ocr: OcrEngine):
        self.ocr = ocr

    def read(self, frame: np.ndarray, layout: FanyaLayout) -> FanyaPageInfo:
        content = layout.content
        lines = self.ocr.read(
            content.crop(frame), offset=(content.x0, content.y0), upscale=1.1
        )
        lines = sort_lines(lines, row_tolerance=max(6, int(0.01 * layout.screen_w)))
        title = self._title(lines, layout)
        tabs = self._tabs(lines, layout)
        active = self._active_tab(frame, tabs)
        completed = self._completed(lines, layout)
        video_rect = self._video_rect(frame, layout)
        # 先在整页 OCR 行里找时间；页面滚动/窗口高度变化时固定 0.79-0.86h
        # 裁剪带可能落不到控制条上。
        time_text, current, total, time_rect = self._time_from_lines(
            lines, video_rect, content
        )
        if not time_text:
            time_text, current, total, time_rect = self._time(frame, layout, video_rect)
        # 播放键就在时间文字左侧同一行。用时间文字定位，避免窗口高度/页面
        # 滚动变化后 0.833*h 落到进度条或视频外，点了没反应。
        play_button: Optional[Rect] = None
        if time_rect is not None:
            # 播放图标约在时间文字左 60px，只框图标本身；框太宽会点到
            # 图标和时间之间的空白/进度条。
            play_button = Rect(
                max(0, time_rect.x0 - 85),
                time_rect.cy - 22,
                max(0, time_rect.x0 - 40),
                time_rect.cy + 22,
            )
            if not (
                content.x0 <= play_button.cx <= content.x1
                and content.y0 <= play_button.cy <= content.y1
            ):
                # 时间必须来自页面内容区；顶部过滤已在 _time_from_lines 做
                play_button = None
        return FanyaPageInfo(
            title=title,
            tabs=tabs,
            active_tab=active,
            video_completed=completed,
            time_text=time_text,
            current=current,
            total=total,
            video_rect=video_rect,
            play_button=play_button,
        )

    @staticmethod
    def _title(lines: list[OcrLine], layout: FanyaLayout) -> str:
        h = layout.screen_h
        candidates = [
            ln
            for ln in lines
            if 0.095 * h <= ln.rect.cy <= 0.21 * h
            and ln.rect.x0 < layout.content.x1 * 0.55
            and len(ln.text.strip()) >= 2
        ]
        # 标题字号明显最大；按行高优先。顶部面包屑/标签栏不是标题。
        candidates.sort(key=lambda ln: (ln.rect.height, len(ln.text)), reverse=True)
        for line in candidates:
            text = line.text.strip()
            if text in UI_TITLE_WORDS:
                continue
            if text.startswith("任务点") or text.startswith("知识点"):
                continue
            return text
        return ""

    @staticmethod
    def _tabs(lines: list[OcrLine], layout: FanyaLayout) -> dict[str, Rect]:
        h = layout.screen_h
        tabs: dict[str, Rect] = {}
        for ln in lines:
            text = ln.text.strip()
            if not (0.16 * h <= ln.rect.cy <= 0.23 * h):
                continue
            if "章节测验" in text or "测验" in text:
                tabs["章节测验"] = ln.rect
            elif "视频" in text:
                tabs["视频"] = ln.rect
        return tabs

    @staticmethod
    def _active_tab(frame: np.ndarray, tabs: dict[str, Rect]) -> str:
        best = ""
        best_score = 0.0
        for name, rect in tabs.items():
            crop = frame[rect.y0 : rect.y1, rect.x0 : rect.x1]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # 选中标签是主题蓝，未选中是灰黑
            mask = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80)
            score = float(mask.mean())
            if score > best_score:
                best_score = score
                best = name
        return best

    @staticmethod
    def _completed(lines: list[OcrLine], layout: FanyaLayout) -> bool:
        h = layout.screen_h
        for ln in lines:
            if not (0.21 * h <= ln.rect.cy <= 0.29 * h):
                continue
            text = ln.text.replace(" ", "")
            if any(word in text for word in COMPLETED_WORDS):
                return True
        return False

    @staticmethod
    def _video_rect(frame: np.ndarray, layout: FanyaLayout) -> Rect:
        c = layout.content
        crop = frame[c.y0 : c.y1, c.x0 : c.x1]
        if crop.size:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mask = (gray < 35).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best = None
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < 0.30 * c.width or h < 0.20 * c.height:
                    continue
                area = int(cv2.contourArea(contour))
                if best is None or area > best[0]:
                    best = (area, x, y, w, h)
            if best is not None:
                _area, x, y, w, h = best
                return Rect(c.x0 + x, c.y0 + y, c.x0 + x + w, c.y0 + y + h)
        # 白屏/加载中/浅色海报时退回固定比例
        return Rect(
            int(c.x1 * 0.0775),
            int(layout.screen_h * 0.188),
            int(c.x1 * 0.973),
            int(layout.screen_h * 0.85),
        )

    @staticmethod
    def _time_from_lines(
        lines: list[OcrLine],
        video_rect: Optional[Rect] = None,
        content: Optional[Rect] = None,
    ) -> tuple[str, Optional[int], Optional[int], Optional[Rect]]:
        for ln in lines:
            in_video = bool(
                video_rect is not None
                and video_rect.x0 - 60 <= ln.rect.cx <= video_rect.x1 + 60
                and video_rect.y0 - 60 <= ln.rect.cy <= video_rect.y1 + 60
            )
            # 播放器时间也可能在视频矩形下方一点；只接受页面下半部分，
            # 明确排除浏览器地址栏/页面顶部的时间。
            in_bottom = bool(
                content is not None
                and ln.rect.cy >= content.y0 + int(0.55 * content.height)
                and ln.rect.cy <= content.y1
            )
            if not in_video and not in_bottom:
                continue
            m = TIME_RANGE_RE.search(ln.text)
            if not m:
                continue
            a, b, c, d = (int(x) for x in m.groups())
            current = a * 60 + b
            total = c * 60 + d
            if total <= 0:
                continue
            return ln.text.strip(), current, total, ln.rect
        return "", None, None, None

    def _time(
        self, frame: np.ndarray, layout: FanyaLayout, video_rect: Rect
    ) -> tuple[str, Optional[int], Optional[int], Optional[Rect]]:
        h = layout.screen_h
        band = Rect(
            video_rect.x0 + 20,
            video_rect.y1 - int(0.12 * video_rect.height),
            video_rect.x1 - 20,
            video_rect.y1 - int(0.01 * video_rect.height),
        )
        lines = self.ocr.read(band.crop(frame), offset=(band.x0, band.y0), upscale=1.6)
        for ln in sort_lines(lines):
            m = TIME_RANGE_RE.search(ln.text)
            if not m:
                continue
            a, b, c, d = (int(x) for x in m.groups())
            current = a * 60 + b
            total = c * 60 + d
            if total <= 0:
                continue
            return ln.text.strip(), current, total, ln.rect
        return "", None, None, None
