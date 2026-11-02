"""泛雅视频中途弹题（ans-videoquiz）识别。

弹题覆盖在视频画面上，DOM 结构（视频 iframe 内）：
    .ans-videoquiz
      .tkTopic_title      判断题 / 单选题 / 多选题
      .tkItem_title       题干
      .tkItem_ul li         A、对   B、错
      #videoquiz-submit    提交
      #videoquiz-continue  继续（提交后出现）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from autolesson.ocr import OcrEngine, sort_lines
from autolesson.types import OcrLine, Rect

log = logging.getLogger("fanya.video_quiz")

KIND_RE = re.compile(r"(判断题|单选题|多选题|单选|多选|判断)")
OPTION_RE = re.compile(r"^\s*([A-H])\s*[、.．:：]\s*(.+?)\s*$")
SUBMIT_WORDS = ("提交",)
CONTINUE_WORDS = ("继续",)


@dataclass
class VideoQuiz:
    kind: str
    question: str
    options: dict[str, str] = field(default_factory=dict)
    option_rects: dict[str, Rect] = field(default_factory=dict)
    submit: Optional[Rect] = None
    continue_button: Optional[Rect] = None
    dialog: Rect = Rect(0, 0, 0, 0)
    raw_lines: list[OcrLine] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.continue_button is not None and not self.options

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.kind} {self.question[:40]} {sorted(self.options)}"


class VideoQuizReader:
    def __init__(self, ocr: OcrEngine, upscale: float = 1.6):
        self.ocr = ocr
        self.upscale = float(upscale)

    def read(
        self,
        frame: np.ndarray,
        content: Rect,
        button_region: Optional[Rect] = None,
    ) -> Optional[VideoQuiz]:
        lines = self.ocr.read(
            content.crop(frame), offset=(content.x0, content.y0), upscale=self.upscale
        )
        lines = sort_lines(lines, row_tolerance=max(6, int(0.008 * content.width)))
        quiz = self.parse(lines, content)
        if quiz is None or not quiz.options:
            # 正常视频页不能每轮都做 2 倍大图 OCR：那会把 2 秒轮询拖到
            # 30 秒，期间控制条隐藏，播放逻辑会误点暂停。
            # 只有当整页 OCR 已出现题型/提交，或视觉上同时有题型+提交
            # 按钮时，才升级到增强 OCR。
            has_quiz_text = any(
                KIND_RE.search(ln.text) or any(w in ln.text for w in SUBMIT_WORDS)
                for ln in lines
            )
            visual = None
            if quiz is None and not has_quiz_text:
                visual = self._visual_quiz(frame, content)
            if quiz is not None or has_quiz_text or visual is not None:
                alt = self._read_enhanced(frame, content)
                if alt is not None and (alt.options or quiz is None):
                    quiz = alt
                if quiz is None:
                    quiz = visual
        if quiz is None:
            return None
        if quiz.submit is None or quiz.continue_button is None:
            # 提交/继续是右上角小号白字蓝底按钮，整卡 OCR 常读不到；
            # 单独放大右下区域再找一次。
            region = button_region or Rect(
                content.x0 + int(0.55 * content.width),
                content.y0 + int(0.30 * content.height),
                content.x1,
                content.y1,
            )
            try:
                extra = self.ocr.read(
                    region.crop(frame),
                    offset=(region.x0, region.y0),
                    upscale=3.0,
                )
            except Exception:  # noqa: BLE001
                extra = []
            if quiz.submit is None:
                quiz.submit = _find_button(extra, SUBMIT_WORDS, region, bottom=True)
            if quiz.continue_button is None:
                quiz.continue_button = _find_button(
                    extra, CONTINUE_WORDS, region, bottom=True, exact=True
                )
        if quiz.submit is None:
            quiz.submit = _find_submit_by_color(frame, content)
        return quiz

    def _read_enhanced(self, frame: np.ndarray, content: Rect) -> Optional[VideoQuiz]:
        crop = content.crop(frame)
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variants = []
        variants.append(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray))
        # 蒙层偏暗时提亮
        variants.append(cv2.convertScaleAbs(gray, alpha=1.4, beta=25))
        best: Optional[VideoQuiz] = None
        for image in variants:
            try:
                lines = self.ocr.read(
                    image,
                    offset=(content.x0, content.y0),
                    upscale=2.0,
                )
            except Exception:  # noqa: BLE001
                continue
            lines = sort_lines(
                lines, row_tolerance=max(6, int(0.008 * content.width))
            )
            candidate = self.parse(lines, content)
            if candidate is None:
                continue
            if candidate.options:
                return candidate
            best = best or candidate
        return best

    def _visual_quiz(
        self, frame: np.ndarray, content: Rect
    ) -> Optional[VideoQuiz]:
        """OCR 完全失败时，用左上题型 + 右下蓝色提交按钮兜底识别。"""
        submit = _find_submit_by_color(frame, content)
        if submit is None:
            return None
        width, height = content.width, content.height
        head = Rect(
            content.x0,
            content.y0,
            content.x0 + int(0.55 * width),
            content.y0 + int(0.28 * height),
        )
        try:
            lines = self.ocr.read(
                head.crop(frame), offset=(head.x0, head.y0), upscale=2.0
            )
        except Exception:  # noqa: BLE001
            lines = []
        kind_line = next(
            (ln for ln in sort_lines(lines) if KIND_RE.search(ln.text) and len(ln.text.strip()) <= 12),
            None,
        )
        if kind_line is None:
            return None
        kind = "判断题" if "判断" in kind_line.text else (
            "多选题" if "多" in kind_line.text else "单选题"
        )
        log.warning(
            "视频弹题 OCR 主路径失败，按题型“%s”+右下提交按钮识别", kind
        )
        return VideoQuiz(
            kind=kind,
            question="",
            submit=submit,
            dialog=content,
            raw_lines=list(lines),
        )

    def parse(self, lines: list[OcrLine], dialog: Rect) -> Optional[VideoQuiz]:
        kind_line = next(
            (ln for ln in lines if KIND_RE.search(ln.text) and len(ln.text.strip()) <= 12),
            None,
        )
        option_lines: list[tuple[OcrLine, str, str]] = []
        for line in lines:
            m = OPTION_RE.match(line.text.strip())
            if m:
                option_lines.append((line, m.group(1).upper(), m.group(2).strip()))
        submit = _find_button(lines, SUBMIT_WORDS, dialog, bottom=True)
        cont = _find_button(lines, CONTINUE_WORDS, dialog, bottom=True, exact=True)
        has_options = len(option_lines) >= 1
        if kind_line is None and not has_options:
            # 提交后只剩“继续”按钮；也要识别出来，否则视频会一直停着。
            if cont is not None:
                return VideoQuiz(
                    kind="",
                    question="",
                    submit=submit,
                    continue_button=cont,
                    dialog=dialog,
                    raw_lines=lines,
                )
            return None
        if not has_options:
            # 只有“判断题”几个字（可能是页面标题）或只有“提交”，不当弹题
            if cont is not None:
                return VideoQuiz(
                    kind="",
                    question="",
                    submit=submit,
                    continue_button=cont,
                    dialog=dialog,
                    raw_lines=lines,
                )
            return None

        options: dict[str, str] = {}
        rects: dict[str, Rect] = {}
        for line, letter, text in option_lines:
            options[letter] = text
            rects[letter] = line.rect

        first_option_y = min((ln.rect.y0 for ln, _, _ in option_lines), default=dialog.y1)
        top_limit = kind_line.rect.y1 if kind_line is not None else dialog.y0 + int(0.12 * dialog.height)
        question_parts = []
        for line in lines:
            text = line.text.strip()
            if not text:
                continue
            if line is kind_line:
                continue
            if line.rect.cy <= top_limit:
                continue
            if line.rect.cy >= first_option_y - 2:
                continue
            if OPTION_RE.match(text):
                continue
            if any(word in text for word in ("提交", "继续", "回答正确", "回答错误")):
                continue
            question_parts.append(text)
        question = " ".join(question_parts).strip()
        kind = "判断题" if kind_line and "判断" in kind_line.text else (
            "多选题" if kind_line and "多" in kind_line.text else "单选题"
        )
        return VideoQuiz(
            kind=kind,
            question=question,
            options=options,
            option_rects=rects,
            submit=submit,
            continue_button=cont,
            dialog=dialog,
            raw_lines=lines,
        )


def _find_button(
    lines: list[OcrLine],
    words: tuple[str, ...],
    dialog: Rect,
    bottom: bool = True,
    exact: bool = False,
) -> Optional[Rect]:
    def matches(ln: OcrLine) -> bool:
        text = ln.text.strip()
        if "中" in text:
            return False
        if exact:
            return text in words
        return any(word in text for word in words)

    candidates = [
        ln
        for ln in lines
        if matches(ln) and (not bottom or ln.rect.cy > dialog.y0 + 0.35 * dialog.height)
    ]
    if not candidates:
        return None
    if bottom:
        line = max(candidates, key=lambda ln: ln.rect.cy)
    else:
        line = min(candidates, key=lambda ln: ln.rect.cy)
    return line.rect.expand(6)


def _find_submit_by_color(frame: np.ndarray, content: Rect) -> Optional[Rect]:
    """右下角蓝色/紫色“提交”按钮；不依赖暗色蒙层 OCR。"""
    crop = content.crop(frame)
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([95, 50, 80], np.uint8), np.array([135, 255, 255], np.uint8)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[tuple[int, tuple[int, int, int, int]]] = None
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < 600:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 40 or h < 18:
            continue
        if x + w < 0.45 * content.width:
            continue
        if y + h < 0.55 * content.height:
            continue
        if best is None or area > best[0]:
            best = (area, (x, y, w, h))
    if best is None:
        return None
    _area, (x, y, w, h) = best
    return Rect(content.x0 + x - 3, content.y0 + y - 3,
                content.x0 + x + w + 3, content.y0 + y + h + 3)
