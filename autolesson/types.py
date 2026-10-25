from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class Rect:
    """左闭右开的像素矩形。"""

    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> "Rect":
        return cls(int(x), int(y), int(x + w), int(y + h))

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def cx(self) -> int:
        return (self.x0 + self.x1) // 2

    @property
    def cy(self) -> int:
        return (self.y0 + self.y1) // 2

    @property
    def center(self) -> tuple[int, int]:
        return self.cx, self.cy

    def crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x0 = min(max(0, self.x0), w)
        y0 = min(max(0, self.y0), h)
        x1 = min(max(0, self.x1), w)
        y1 = min(max(0, self.y1), h)
        return img[y0:y1, x0:x1]

    def offset(self, dx: int, dy: int) -> "Rect":
        return Rect(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def expand(self, pad: int) -> "Rect":
        return Rect(self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad)

    def clamp(self, w: int, h: int) -> "Rect":
        return Rect(
            min(max(0, self.x0), w),
            min(max(0, self.y0), h),
            min(max(0, self.x1), w),
            min(max(0, self.y1), h),
        )

    def union(self, other: "Rect") -> "Rect":
        return Rect(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def iou(self, other: "Rect") -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if inter <= 0:
            return 0.0
        return inter / float(self.area + other.area - inter)

    def as_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    def __str__(self) -> str:  # pragma: no cover - 日志辅助
        return f"({self.x0},{self.y0},{self.x1},{self.y1})"


@dataclass
class OcrLine:
    text: str
    rect: Rect
    score: float = 1.0

    @property
    def cx(self) -> int:
        return self.rect.cx

    @property
    def cy(self) -> int:
        return self.rect.cy

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.text!r}@{self.rect}[{self.score:.2f}]"


STATUS_COMPLETED = "completed"
STATUS_IN_PROGRESS = "in_progress"
STATUS_NOT_STARTED = "not_started"


def normalize_title(text: str) -> str:
    """标题归一化，用于跨帧匹配。"""
    text = re.sub(r"\s+", "", text or "")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("：", ":").replace("，", ",")
    return text


@dataclass
class Lesson:
    """右侧目录中的一个视频条目（不含平时测试）。"""

    index: Optional[str]
    title: str
    duration: int
    status: str
    title_rect: Rect
    row: Rect
    percent: Optional[int] = None

    @property
    def key(self) -> str:
        return f"{self.index or '-'}|{normalize_title(self.title)}"

    @property
    def title_key(self) -> str:
        return normalize_title(self.title)

    @property
    def completed(self) -> bool:
        return self.status == STATUS_COMPLETED

    def duration_text(self) -> str:
        m, s = divmod(max(0, int(self.duration)), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def __str__(self) -> str:  # pragma: no cover
        pct = f" {self.percent}%" if self.percent is not None else ""
        return f"{self.index or '-'} {self.title} [{self.duration_text()}] {self.status}{pct}"


@dataclass
class ProgressBar:
    y: int
    x_right: int
    playhead_x: int
    ratio: float
    rows: tuple[int, ...] = ()


@dataclass
class PlaybackState:
    current: Optional[float]
    total: Optional[float]
    ratio: Optional[float]
    playing: Optional[bool]
    bar_y: int
    play_button: Optional[Rect]
    time_rect: Optional[Rect] = None
    raw_time: str = ""

    @property
    def percent(self) -> Optional[float]:
        if self.ratio is None:
            return None
        return max(0.0, min(1.0, self.ratio)) * 100.0


@dataclass
class Overlay:
    kind: str
    rect: Rect
    text: str = ""
    header: str = ""
    footer: str = ""
    raw_text: str = ""


@dataclass
class QuizQuestion:
    kind: str
    text: str
    options: dict[str, str]
    option_rects: dict[str, Rect]
    dialog: Rect
    header: str = ""
    needs_vision: bool = False
    ocr_score: float = 1.0
    raw_lines: list[OcrLine] = field(default_factory=list)

    @property
    def letters(self) -> list[str]:
        return sorted(self.options.keys())

    def hash_text(self) -> str:
        import hashlib

        payload = self.text + "\n" + "\n".join(f"{k}.{self.options[k]}" for k in self.letters)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def prompt_text(self) -> str:
        lines = [f"题型：{self.kind or '未知'}"] if self.kind else []
        lines.append(f"题目：{self.text}")
        lines.append("选项：")
        image_option = False
        for k in self.letters:
            value = (self.options.get(k) or "").strip()
            if value:
                lines.append(f"{k}. {value}")
            else:
                image_option = True
                lines.append(f"{k}. （图片选项，见弹窗截图）")
        if image_option:
            lines.append("说明：部分选项正文是图片，请结合截图按选项从上到下的 A/B/C/D 顺序判断。")
        return "\n".join(lines)


@dataclass
class LLMAnswer:
    letters: list[str]
    confidence: float
    reason: str = ""
    source: str = "text"
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class AutoLessonError(Exception):
    """可预期错误基类。"""


class FatalError(AutoLessonError):
    """必须立即停止并通知用户的故障。"""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason if not detail else f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class LessonError(AutoLessonError):
    """单个视频处理失败，可尝试重试。"""


class UserAbort(AutoLessonError):
    """用户主动中止（Ctrl+C 或鼠标移入安全角）。"""
