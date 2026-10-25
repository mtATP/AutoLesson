"""自动练习（平时测试）页面处理。

安全约定：
- 只点击“暂存作业/暂存/保存”，代码里不存在点击“提交作业”的路径；
- 提交确认弹窗一律取消；
- 已提交/无法再次提交/已截止的页面直接跳过；
- 已经选中的题目不再作答，只补未作答题目；
- 题目截图交给多模态大模型作答。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from . import vision
from .config import Config
from .llm import LLMConfigError, LLMError, random_fallback, report_llm_fallback
from .layout import Layout
from .ocr import OcrEngine, sort_lines
from .panel import PanelNavigator, PanelScanner
from .types import OcrLine, Rect, STATUS_COMPLETED
from .quiz import parse_quiz

log = logging.getLogger("autolesson.practice")

TEST_WORDS = ("平时测试", "平时测验", "章节测试")
OPTION_RE = re.compile(r"^[^\w\u4e00-\u9fff]*([A-D])\s*[.．、]\s*(.*)$")
QUESTION_RE = re.compile(r"^\s*(\d{1,2})\s*[.．、]")
PART_RE = re.compile(r"第\s*\d+\s*部分")
CHAPTER_RE = re.compile(r"第\s*[一二三四五六七八九十百零〇\d]+\s*章")
COUNT_RE = re.compile(r"(?:总题数|题目数)\s*[:：]?\s*(\d{1,3})")
SUBMITTED_WORDS = (
    "作业提交成功",
    "提交成功",
    "已提交",
    "无法再次提交",
    "提交后无法修改",
    "再测一次",
    "本次测试成绩",
    "作业已提交",
    "已截止",
    # 第二种完成界面：得分/评语（不把普通“AI解析”按钮当结果页）
    "本题的得分",
    "评语",
    "总得分",
    "查看解析",
    "考试结果",
    "得 分",
)


@dataclass
class PracticeEntry:
    chapter: str
    title: str
    rect: Rect
    key: str
    completed: bool = False

    def __str__(self) -> str:
        mark = " [已完成]" if self.completed else ""
        return f"{self.chapter or '?'} {self.title}{mark} @ {self.rect.center}"


def chapter_key(text: str) -> str:
    m = CHAPTER_RE.search(text or "")
    return re.sub(r"\s+", "", m.group(0)) if m else ""


class PracticeScanner:
    """在右侧目录里寻找“平时测试”行及其所属章节。"""

    def __init__(self, ocr: OcrEngine, cfg: Config):
        self.ocr = ocr
        self.cfg = cfg

    def scan(self, frame: np.ndarray, layout: Layout) -> list[PracticeEntry]:
        panel = layout.panel
        if panel.width < 80 or panel.height < 80:
            return []
        lines = self.ocr.read(
            panel.crop(frame),
            offset=(panel.x0, panel.y0),
            upscale=float(self.cfg.panel.ocr_upscale or 1.0),
        )
        return self.parse(lines, frame, layout)

    def parse(
        self, lines: list[OcrLine], frame: np.ndarray, layout: Layout
    ) -> list[PracticeEntry]:
        """解析已经 OCR 好的目录文字，供当前视口和拼接长图共用。"""
        panel = layout.panel
        lines = sort_lines(lines, row_tolerance=max(6, int(0.012 * layout.screen_w)))
        heights = [ln.rect.height for ln in lines if ln.rect.height >= 8]
        font_h = int(np.median(heights)) if heights else 18
        entries: list[PracticeEntry] = []
        for line in lines:
            if line.rect.y0 < panel.y0 or line.rect.y1 > min(layout.screen_h, panel.y1):
                continue
            text = line.text.strip()
            if not any(w in text for w in TEST_WORDS):
                continue
            above = [ln for ln in lines if ln.cy < line.cy - 1]
            chapter = ""
            headers = [ln for ln in above if CHAPTER_RE.search(ln.text)]
            if headers:
                chapter = chapter_key(headers[-1].text)
            if not chapter:
                indexes = [ln for ln in above if re.match(r"^\s*(\d{1,2})\s*[.．]\s*(\d{1,2})\s*$", ln.text)]
                if indexes:
                    chapter = index = indexes[-1].text.strip().split(".")[0].split("．")[0]
            key = chapter or f"row:{line.rect.y0}:{line.text}"
            # 平时测试完成会像视频一样打蓝勾；必须在点击前判断，避免进入结果页
            row_top = line.rect.cy - int(1.2 * font_h)
            row_bottom = line.rect.cy + int(1.2 * font_h)
            zone = Rect(
                panel.x1 - int(self.cfg.panel.right_icon_width),
                row_top,
                panel.x1,
                row_bottom,
            )
            status, _icon = vision.find_status_icon(
                frame, zone, row_top, row_bottom, font_h, anchor_y=line.rect.cy
            )
            entries.append(
                PracticeEntry(
                    chapter=chapter,
                    title=text,
                    rect=line.rect,
                    key=key,
                    completed=(status == STATUS_COMPLETED),
                )
            )
        return entries


class PracticePage:
    """处理当前打开的一个练习/测试标签页。"""

    def __init__(
        self,
        ocr: OcrEngine,
        cfg: Config,
        mouse,
        screen,
        llm,
        state,
        logger: Any = None,
        stop_event=None,
        notifier=None,
        pause_event=None,
    ):
        self.ocr = ocr
        self.cfg = cfg
        self.mouse = mouse
        self.screen = screen
        self.llm = llm
        self.state = state
        self.log = logger or log
        self.stop_event = stop_event
        self.notifier = notifier
        self.pause_event = pause_event
        self.answer_mode = str(getattr(cfg.practice, "answer_mode", "image"))

    # ------------------------------------------------------------------ 基础
    def _stopped(self) -> bool:
        if self.pause_event is not None:
            while self.pause_event.is_set():
                if self.stop_event is not None and self.stop_event.is_set():
                    return True
                time.sleep(0.15)
        return bool(self.stop_event is not None and self.stop_event.is_set())

    def _read(self, frame: np.ndarray, upscale: float = 1.0) -> list[OcrLine]:
        lines = self.ocr.read(frame, upscale=upscale)
        return sort_lines(lines, row_tolerance=max(6, int(0.008 * frame.shape[1])))

    @staticmethod
    def _text(lines: list[OcrLine]) -> str:
        return "\n".join(ln.text for ln in lines)

    @staticmethod
    def _find(lines: list[OcrLine], needles, exclude=(), exact: bool = False) -> Optional[OcrLine]:
        for ln in lines:
            text = ln.text.strip()
            if exact:
                if text in needles:
                    return ln
                continue
            if any(n in text for n in needles) and not any(x in text for x in exclude):
                return ln
        return None

    def _find_save(self, lines: list[OcrLine]) -> Optional[OcrLine]:
        # 只认暂存；绝不把“提交/保存并提交”当保存按钮
        return self._find(lines, ("暂存作业", "暂存"), exclude=("提交",))

    def _find_submit(self, lines: list[OcrLine]) -> Optional[OcrLine]:
        return self._find(lines, ("提交作业", "提交"), exclude=("暂存",))

    @staticmethod
    def _is_submitted_text(text: str) -> bool:
        return any(w in text for w in SUBMITTED_WORDS)

    def _chapter(self, text: str) -> str:
        return chapter_key(text)

    # ------------------------------------------------------------------ 页面状态
    def wait_for_page(self, timeout: Optional[float] = None) -> bool:
        if timeout is None:
            timeout = float(getattr(self.cfg.practice, "page_timeout", 30.0))
        deadline = time.monotonic() + max(1.0, timeout)
        while time.monotonic() < deadline:
            if self._stopped():
                return False
            frame = self.screen.grab()
            lines = self._read(frame)
            text = self._text(lines)
            # 可作答练习看“暂存作业”；已提交结果页看得分/解析标记
            if self._find_save(lines) is not None or self._is_submitted_text(text):
                return True
            time.sleep(0.8)
        return False

    def _question_count(
        self,
        lines: list[OcrLine],
        text: str,
        frame: Optional[np.ndarray] = None,
    ) -> int:
        # “题目数 3 / 总题数: 3”优先，绝不能因为 OCR 暂时没读到就默认 10
        m = COUNT_RE.search(text)
        if m:
            return max(1, min(100, int(m.group(1))))
        # 其次看右侧答题卡最大数字
        if frame is not None:
            nums = self._answer_card_numbers(frame, count=50)
            if nums:
                return max(nums)
        return int(getattr(self.cfg.practice, "default_questions", 10))

    def _answer_card_numbers(
        self,
        frame: Optional[np.ndarray],
        count: int,
        lines: Optional[list[OcrLine]] = None,
    ) -> dict[int, Rect]:
        if frame is None:
            return {}
        h, w = frame.shape[:2]
        # 排除屏幕顶部：置顶小条/浏览器标签栏也可能被 OCR 成数字，
        # 误点会把 GUI 小条唤到前台。答题卡数字在右侧且位于页头下方。
        x0 = int(0.72 * w)
        y0 = int(0.10 * h)
        crop = frame[y0:, x0:]
        try:
            card_lines = self.ocr.read(crop, offset=(x0, y0), upscale=1.8)
        except Exception:
            return {}
        numbers: dict[int, Rect] = {}
        for ln in card_lines:
            text = ln.text.strip()
            if text.isdigit() and 1 <= int(text) <= count:
                numbers[int(text)] = ln.rect
        # 答题卡通常是两行（1-5 / 6-10）。必须先按行分组，再在行内等距补数字；
        # 旧逻辑按整体 y 差判断“竖排”，两行卡片会被一条竖线插值，点击坐标可能跑到
        # 顶部 GUI 小条，正是“点第 2 题却唤出 GUI”的来源。
        if len(numbers) < 2:
            return numbers
        widths = [r.width for r in numbers.values()]
        heights = [r.height for r in numbers.values()]
        w = int(np.median(widths))
        hh = int(np.median(heights))
        items = sorted(((n, r.cx, r.cy) for n, r in numbers.items()), key=lambda x: x[2])
        rows: list[list[tuple[int, int, int]]] = []
        for item in items:
            placed = False
            for row in rows:
                row_y = float(np.median([x[2] for x in row]))
                if abs(item[2] - row_y) <= max(8, 1.5 * hh):
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])

        def row_fill(row: list[tuple[int, int, int]]) -> None:
            known = sorted(row, key=lambda x: x[0])
            if len(known) < 2:
                return
            nums = np.array([x[0] for x in known], dtype=float)
            xs = np.array([x[1] for x in known], dtype=float)
            ys = np.array([x[2] for x in known], dtype=float)
            # 线性拟合题号 -> x；只补这一行范围内的题号
            if np.ptp(nums) >= 1:
                slope, intercept = np.polyfit(nums, xs, 1)
                for n in range(int(nums.min()), int(nums.max()) + 1):
                    if n in numbers:
                        continue
                    x = int(round(intercept + slope * n))
                    y = int(round(float(np.median(ys))))
                    numbers[n] = Rect(x - w // 2, y - hh // 2, x + w // 2, y + hh // 2)

        for row in rows:
            row_fill(row)
        # 行首/行尾缺号（例如 6 被高亮漏读）用最近已知行拟合外推
        for n in range(1, count + 1):
            if n in numbers:
                continue
            # 选“数字上最接近”的答题卡行；单行卡片就补全整行，
            # 两行卡片则分别向 1..5 / 6..10 两端外推。
            if not rows:
                continue
            row = min(
                rows,
                key=lambda r: min(abs(n - x[0]) for x in r) if r else 10**9,
            )
            known = sorted(row, key=lambda x: x[0])
            if not known:
                continue
            nums = np.array([x[0] for x in known], dtype=float)
            xs = np.array([x[1] for x in known], dtype=float)
            ys = np.array([x[2] for x in known], dtype=float)
            if len(known) >= 2 and np.ptp(nums) >= 1:
                slope, intercept = np.polyfit(nums, xs, 1)
                x = int(round(intercept + slope * n))
            else:
                step = int(np.median([r.width for r in numbers.values()])) + 12
                x = int(round(xs[0] + (n - nums[0]) * step))
            y = int(round(float(np.median(ys))))
            numbers[n] = Rect(x - w // 2, y - hh // 2, x + w // 2, y + hh // 2)
        return numbers
        return numbers

    def _card_is_blue(self, frame: np.ndarray, rect: Rect) -> bool:
        """答题卡题号背景为蓝色 = 当前题或已完成题。"""
        try:
            h = max(8, rect.height)
            crop = Rect(
                rect.x0 - h, rect.y0 - h // 2, rect.x1 + h, rect.y1 + h // 2
            ).crop(frame)
            if crop.size == 0:
                return False
            mask = vision.blue_mask(crop)
            return float(mask.mean()) > 0.12
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _is_answer_card_rect(frame: np.ndarray, rect: Rect) -> bool:
        h, w = frame.shape[:2]
        return rect.cx > 0.72 * w and rect.y0 > 0.10 * h

    def _first_unanswered(self, frame: np.ndarray, count: int) -> int:
        numbers = self._answer_card_numbers(frame, count)
        current = self._current_question_number(frame)
        completed: set[int] = set()
        if current is not None:
            try:
                q = self._read_question_once(frame)
                if q is not None and self._selected_letters(frame, q):
                    completed.add(current)
            except Exception:  # noqa: BLE001
                pass
        for n in range(1, count + 1):
            rect = numbers.get(n)
            if rect is None:
                continue
            if n != current and self._card_is_blue(frame, rect):
                completed.add(n)
        for n in range(1, count + 1):
            if n not in completed:
                return n
        return 1

    def _find_next_button(self, frame: np.ndarray, lines: Optional[list[OcrLine]] = None) -> Optional[OcrLine]:
        lines = self._read(frame) if lines is None else lines
        nxt = self._find(
            lines,
            ("下一题",),
            exclude=("答题后", "保存答案", "全部试题", "提交", "避免答题记录"),
        )
        if nxt is not None:
            return nxt
        h, w = frame.shape[:2]
        bottom = Rect(int(0.35 * w), int(0.58 * h), int(0.70 * w), int(0.82 * h))
        try:
            more = self.ocr.read(
                bottom.crop(frame), offset=(bottom.x0, bottom.y0), upscale=2.0
            )
        except Exception:  # noqa: BLE001
            return None
        return self._find(
            more,
            ("下一题",),
            exclude=("答题后", "保存答案", "全部试题", "提交", "避免答题记录"),
        )

    @staticmethod
    def _stitch_images(images: list[np.ndarray]) -> Optional[np.ndarray]:
        if not images:
            return None
        merged = images[0].copy()
        for new in images[1:]:
            if new is None or new.size == 0:
                continue
            if merged.shape[1] != new.shape[1]:
                continue
            tail_h = max(30, min(merged.shape[0] // 2, int(merged.shape[0] * 0.22)))
            tail = merged[-tail_h:, :]
            try:
                res = cv2.matchTemplate(new, tail, cv2.TM_CCOEFF_NORMED)
                _min, maxv, _minloc, maxloc = cv2.minMaxLoc(res)
            except cv2.error:
                maxv, maxloc = 0.0, (0, 0)
            if maxv >= 0.55:
                rest = new[int(maxloc[1]) + tail_h :, :]
                if rest.size:
                    merged = np.vstack([merged, rest])
            else:
                merged = np.vstack([merged, new])
        return merged

    # ------------------------------------------------------------------ 题目解析
    def _question_dialog(
        self, frame: np.ndarray, lines: list[OcrLine]
    ) -> tuple[Optional[Rect], Optional[OcrLine]]:
        h, w = frame.shape[:2]
        parts = [ln for ln in lines if PART_RE.search(ln.text)]
        # 答题卡里也有“第1部分”，取最靠左的正文分区标题
        part = min(parts, key=lambda ln: ln.rect.x0) if parts else None
        question = next(
            (
                ln
                for ln in lines
                if QUESTION_RE.match(ln.text)
                and ("【" in ln.text or "分" in ln.text or ln.rect.y0 > 0.15 * h)
            ),
            None,
        )
        options = [ln for ln in lines if OPTION_RE.match(ln.text)]
        if question is None and options:
            question = min(options, key=lambda ln: ln.rect.y0)
        if part is None and question is None:
            return None, None

        pager = [
            ln
            for ln in lines
            if ln.rect.y0 > 0.62 * h
            and any(k in ln.text for k in ("下一题", "上一题", "保存"))
            and "提交" not in ln.text
        ]
        xs = [ln.rect.x0 for ln in (part, question, *options) if ln is not None]
        xe = [ln.rect.x1 for ln in (part, question, *options) if ln is not None]
        if not xs or not xe:
            return None, question
        x0 = max(0, min(xs) - int(0.02 * w))
        # 右侧答题卡不是题目正文；覆盖正文宽度但避开右侧答题卡
        x1 = min(w, max(max(xe) + int(0.04 * w), int(0.85 * w)))
        y0 = int((part or question).rect.y0) - 12
        if pager:
            y1 = min(ln.rect.y0 for ln in pager) - 12
        else:
            y1 = min(h, max(ln.rect.y1 for ln in options or [question]) + 50)
        if y1 <= y0 + 40 or x1 <= x0 + 80:
            return None, question
        return Rect(x0, max(0, y0), x1, min(h, y1)), question

    def _page_scroll(self, direction: int = -1, pixels: int = 300) -> bool:
        """滚动整个答题页面，用于把视口外的 C/D 选项露出来。"""
        frame = self.screen.grab()
        h, w = frame.shape[:2]
        before = vision.image_signature(frame)
        remaining = abs(int(pixels))
        x, y = int(0.45 * w), int(0.50 * h)
        while remaining > 0:
            chunk = min(60, remaining)
            self.mouse.scroll_at(x, y, int(direction) * chunk)
            remaining -= chunk
            time.sleep(0.03)
        time.sleep(0.35)
        after = vision.image_signature(self.screen.grab())
        return vision.signature_distance(before, after) > 40

    def _scroll_page_to_top(self) -> None:
        for _ in range(5):
            if not self._page_scroll(+1, 600):
                break

    def _read_question_once(self, frame: np.ndarray) -> Optional[Any]:
        lines = self._read(frame)
        dialog, _qline = self._question_dialog(frame, lines)
        if dialog is None:
            return None
        upscale = 1.5 if dialog.width >= 900 else 1.8
        try:
            qlines = self.ocr.read(
                dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=upscale
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("练习题目 OCR 失败: %s", exc)
            return None
        q = parse_quiz(qlines, dialog, frame, self.answer_mode)
        self._recover_option_a(q, qlines, dialog)
        self._recover_option_letters(q, lines, dialog)
        return q

    def read_question(self, frame: Optional[np.ndarray] = None) -> Optional[Any]:
        """读取题目；若选项在视口外，向下滚动并合并 A/B/C/D。"""
        frame = self.screen.grab() if frame is None else frame
        q = self._read_question_once(frame)
        if q is None:
            return None
        if "判断" in (q.kind or ""):
            needed = 2
        elif "多选" in (q.kind or ""):
            needed = 5
        else:
            needed = 4
        for _ in range(4):
            if len(q.options) >= needed:
                break
            if not self._page_scroll(-1, 360):
                break
            page = self._read_question_once(self.screen.grab())
            if page is None:
                break
            before = len(q.options)
            for letter, rect in page.option_rects.items():
                if letter not in q.option_rects:
                    q.options[letter] = page.options.get(letter, letter)
                    q.option_rects[letter] = rect
            if len(q.options) == before and len(page.options) <= len(q.options):
                break
        return q

    def _locate_option(self, letter: str) -> Optional[Rect]:
        """从页面顶部向下滚动，直到目标选项完整可见。"""
        self._scroll_page_to_top()
        for _ in range(8):
            frame = self.screen.grab()
            q = self._read_question_once(frame)
            rect = q.option_rects.get(letter) if q is not None else None
            if rect is not None and rect.y0 > 40 and rect.y1 < frame.shape[0] - 40:
                return rect
            if not self._page_scroll(-1, 360):
                break
        return None

    def _recover_option_letters(self, q, lines: list[OcrLine], dialog: Rect) -> None:
        """数学公式选项的 OCR 常只识别出字母，补全 A/B/C/D 选项行。"""
        if q is None:
            return
        for ln in lines:
            if not dialog.contains(ln.cx, ln.cy):
                continue
            m = OPTION_RE.match(ln.text)
            if not m:
                continue
            letter = m.group(1)
            if letter in q.option_rects:
                continue
            text = m.group(2).strip()
            q.options.setdefault(letter, text or letter)
            q.option_rects.setdefault(letter, ln.rect)

    def _recover_option_a(self, q, qlines: list[OcrLine], dialog: Rect) -> None:
        """页面里 A. 的字母经常和选项文字/公式分离，OCR 会漏掉 A。"""
        if q is None or "A" in q.options:
            return
        b = q.option_rects.get("B")
        if b is None:
            return
        font_h = max(12, b.height)
        candidates = []
        for ln in qlines:
            text = ln.text.strip()
            if not text:
                continue
            if re.match(r"^\s*[B-D]\s*[.．、]", text):
                continue
            if ln.rect.y1 > b.y0 - 4:
                continue
            if ln.rect.y0 < b.y0 - 3.2 * font_h:
                continue
            if ln.rect.x0 < b.x0 - 0.08 * dialog.width:
                continue
            if any(w in text for w in ("下一题", "上一题", "提交", "暂存")):
                continue
            candidates.append(ln)
        if not candidates:
            return
        a = max(candidates, key=lambda ln: ln.rect.y1)
        row = [ln for ln in candidates if abs(ln.cy - a.cy) <= 1.3 * font_h]
        rect = row[0].rect
        for ln in row[1:]:
            rect = rect.union(ln.rect)
        q.options["A"] = " ".join(ln.text.strip() for ln in row) or "A"
        q.option_rects["A"] = rect

    def _option_selected(self, frame: np.ndarray, rect: Rect) -> bool:
        # 只检查选项文字左侧的圆点/复选框：选项字母本身也可能是蓝色的，
        # 若把整行文字算进去会把所有选项都误判为“已选”。
        font_h = max(10, rect.height)
        area = Rect(
            rect.x0 - int(2.4 * font_h),
            rect.cy - int(0.7 * font_h),
            rect.x0 - int(0.15 * font_h),
            rect.cy + int(0.7 * font_h),
        )
        crop = area.crop(frame)
        if crop.size == 0:
            return False
        mask = vision.blue_mask(crop)
        return float(mask.mean()) > 0.02

    def _selected_letters(self, frame: np.ndarray, q) -> list[str]:
        return [letter for letter, rect in q.option_rects.items() if self._option_selected(frame, rect)]

    # ------------------------------------------------------------------ 作答
    def _image_viewer_open(self, frame: Optional[np.ndarray] = None) -> bool:
        """题目图片被点开后会出现底部 1:1 缩放工具栏，遮住题目。"""
        frame = self.screen.grab() if frame is None else frame
        h = frame.shape[0]
        # 只 OCR 底部工具条区域，避免每题多做一次全屏 OCR
        top = int(0.75 * h)
        strip = frame[top:, :]
        try:
            lines = self.ocr.read(strip, offset=(0, top), upscale=1.5)
        except Exception:  # noqa: BLE001
            return False
        return any(
            ("1:1" in ln.text or "1：1" in ln.text) and ln.cy > int(0.78 * h)
            for ln in lines
        )

    def close_image_viewer(self) -> bool:
        frame = self.screen.grab()
        if not self._image_viewer_open(frame):
            return False
        self.log.warning("检测到题目图片查看器，按 Esc 关闭")
        self.mouse.hotkey("esc")
        time.sleep(0.7)
        if self._image_viewer_open():
            # 兜底：点击图片查看器右上角的 X
            self.mouse.click(
                int(0.985 * frame.shape[1]),
                int(0.055 * frame.shape[0]),
                "关闭图片查看器",
            )
            time.sleep(0.7)
        return True

    def apply_answer(self, frame: np.ndarray, q, letters: list[str]) -> bool:
        selected_ok: set[str] = set()
        for letter in letters:
            # 选项可能在视口外：从题目顶部重新找，找到再点
            rect = self._locate_option(letter)
            if rect is None:
                self.log.error("练习中找不到选项 %s", letter)
                return False
            q.option_rects[letter] = rect
            frame = self.screen.grab()
            if self._option_selected(frame, rect):
                selected_ok.add(letter)
                continue
            font_h = max(10, rect.height)
            # 先点选项左侧圆点/复选框；绝不点文字中心，否则含图片的题目会打开图片查看器
            self.mouse.click(
                rect.x0 - int(0.9 * font_h), rect.cy, f"选择 {letter}(圆点)"
            )
            time.sleep(0.35)
            frame = self.screen.grab()
            if not self._option_selected(frame, rect):
                self.mouse.click(
                    rect.x0 - int(1.8 * font_h), rect.cy, f"选择 {letter}(选项框)"
                )
                time.sleep(0.35)
                frame = self.screen.grab()
            if self._option_selected(frame, rect):
                selected_ok.add(letter)
        missing = [x for x in letters if x not in selected_ok]
        if missing:
            self.log.warning("选项 %s 可能未选中", missing)
            return False
        return True

    def _collect_full_question(self, q) -> tuple[str, list[np.ndarray]]:
        """慢速扫完整页，收集长代码/长题的多屏截图与全文，避免只截首屏。"""
        self._scroll_page_to_top()
        frame = self.screen.grab()
        h, w = frame.shape[:2]
        # 只取正文列，避开右侧答题卡；保留整页高度
        content = Rect(int(0.10 * w), int(0.05 * h), int(0.72 * w), int(0.96 * h))
        quick = self._read_question_once(frame)
        if (
            quick is not None
            and len(quick.options) >= 2
            and self._find_next_button(frame) is not None
        ):
            self.log.info("题目首屏完整，单图提交")
            return quick.text, [content.crop(frame)]
        images: list[np.ndarray] = []
        extra: list[str] = []
        seen: set[str] = set()
        for _ in range(8):
            frame = self.screen.grab()
            body = content.crop(frame)
            if body.size:
                images.append(body)
            try:
                lines = self.ocr.read(
                    body, offset=(content.x0, content.y0), upscale=1.3
                )
            except Exception as exc:  # noqa: BLE001
                self.log.debug("长题正文 OCR 失败: %s", exc)
                lines = []
            for ln in lines:
                text = ln.text.strip()
                if not text or text in seen:
                    continue
                if any(k in text for k in ("答题后", "Copyright", "暂存作业", "提交作业")):
                    continue
                seen.add(text)
                extra.append(text)
            if not self._page_scroll(-1, 360):
                break
        self._scroll_page_to_top()
        if extra:
            # 原题干在前，后面补整页 OCR 出来的代码/说明
            q.text = (q.text + "\n" + "\n".join(extra)).strip()
        self.log.info("长题采集: %d 屏，文本 %d 行", len(images), len(extra))
        return q.text, images

    def answer_current(self) -> str:
        self.close_image_viewer()
        frame = self.screen.grab()
        q = self.read_question(frame)
        if q is None or not q.text or len(q.options) < 2:
            self.log.error("练习题目识别不完整，跳过本题")
            return "failed"

        # 正确率优先：每次都从题目顶部慢速扫描，拿完整代码/长题文本和多屏截图
        full_text, images = self._collect_full_question(q)
        if full_text:
            q.text = full_text
        if not images:
            images = [q.dialog.crop(self.screen.grab())]
        stitched = self._stitch_images(images)
        if stitched is None:
            stitched = images[0]
        complete = len(images) <= 1
        self._scroll_page_to_top()
        frame = self.screen.grab()
        live = self._read_question_once(frame)
        selected = self._selected_letters(frame, live or q)
        if selected:
            self.log.info("题目已作答 %s，跳过不重复作答", selected)
            return "skipped"
        self.log.info(
            "练习题目: %s | 选项: %s | 完整=%s 屏数=%d",
            q.text[:120],
            {k: v[:24] for k, v in q.options.items()},
            complete,
            len(images),
        )

        if complete:
            # 完整题：单图 → 图片重试 → 纯文本
            modes = [("image", "单图"), ("image", "单图重试"), ("text", "纯文本")]
        else:
            # 不完整：拼接图 → 拼接重试 → 不拼接多图 → 纯文本
            modes = [
                ("image", "拼接图"),
                ("image", "拼接图重试"),
                ("multi", "多图"),
                ("text", "纯文本"),
            ]

        answer = None
        last_error: Optional[Exception] = None
        for kind, label in modes:
            try:
                if kind == "image":
                    answer = self.llm.answer(
                        q, images=[stitched], auto_switch=False
                    )
                elif kind == "multi":
                    answer = self.llm.answer(q, images=images, auto_switch=False)
                else:
                    answer = self.llm.answer(
                        q, image_bgr=None, auto_switch=False
                    )
                if answer is not None and answer.letters:
                    self.log.info("大模型作答模式: %s", label)
                    break
                last_error = LLMError("大模型返回空答案")
            except LLMConfigError:
                raise
            except LLMError as exc:
                last_error = exc
                self.log.warning("大模型%s失败: %s", label, exc)
            answer = None

        if answer is None or not answer.letters:
            message = (
                f"题目: {q.text[:200]}\n"
                "大模型在单图/重试/多图/纯文本模式下均未给出有效答案；"
                "练习模式不使用随机兜底，已跳过本题、不选择任何选项，等待人工复核。"
            )
            self.log.error(message)
            if self.notifier is not None:
                try:
                    self.notifier.send(
                        "练习题目跳过（大模型无有效答案）", message, "warning", force=True
                    )
                except Exception:  # noqa: BLE001
                    pass
            return "skipped"
        self.log.info(
            "大模型答案: %s (置信度 %.2f, 来源 %s) %s",
            answer.letters,
            answer.confidence,
            answer.source,
            answer.reason[:80],
        )
        if not self.apply_answer(frame, q, answer.letters):
            return "failed"
        return "answered"

    def _current_question_number(self, frame: Optional[np.ndarray] = None) -> Optional[int]:
        """用正文左侧的题号判断当前题；答题卡/置顶数字不算。"""
        frame = self.screen.grab() if frame is None else frame
        h, w = frame.shape[:2]
        lines = self._read(frame)
        candidates = []
        for ln in lines:
            if ln.rect.cx > 0.72 * w or ln.rect.y0 < 0.08 * h:
                continue
            m = QUESTION_RE.match(ln.text)
            if not m:
                continue
            num = int(m.group(1))
            if 1 <= num <= 100:
                candidates.append((ln.rect.y0, ln.rect.x0, num))
        if not candidates:
            return None
        # 取正文最上方、最靠左的题号，避开答题卡里的数字
        return min(candidates)[2]

    def next_question(self, count: int, index: int) -> bool:
        self.close_image_viewer()
        frame = self.screen.grab()
        before = self._current_question_number(frame)
        lines = self._read(frame)
        # 底部说明文字里也有“下一题”，必须排除，否则会点到提示语而不是按钮
        nxt = self._find(
            lines,
            ("下一题",),
            exclude=("答题后", "保存答案", "全部试题", "提交", "避免答题记录"),
        )
        if nxt is None:
            nxt = self._find(lines, ("保存",), exclude=("提交", "暂存", "答题后", "全部试题"))
        if nxt is not None:
            # OCR 框是按钮文字的紧框；从文字中心逐步向下试，直到题号变化。
            # 0.85 倍字高大约落在按钮下半部，避免鼠标停在按钮上边缘。
            # 文字框中心本身就在按钮内部；先点中心，再小范围上下试。
            # 不要用 0.85 倍字高，按钮高约 3 倍字高以下，过大会点到按钮下边缘外。
            for ratio in (0.15, 0.35, 0.0, -0.10):
                y = nxt.rect.cy + int(ratio * nxt.rect.height)
                self.mouse.click(nxt.rect.cx, y, "下一题/保存")
                time.sleep(float(getattr(self.cfg.practice, "question_wait", 0.8)))
                after = self._current_question_number()
                if before is not None and after is not None and after != before:
                    return True
                self.log.warning(
                    "点击下一题后题号未变化（%s -> %s），下移落点重试",
                    before,
                    after,
                )
        else:
            # 全屏 OCR 偶尔会漏掉蓝色按钮文字；单独放大底部按钮带再找一次
            h, w = frame.shape[:2]
            bottom = Rect(int(0.35 * w), int(0.58 * h), int(0.70 * w), int(0.82 * h))
            try:
                more = self.ocr.read(
                    bottom.crop(frame), offset=(bottom.x0, bottom.y0), upscale=2.0
                )
                nxt = self._find(
                    more,
                    ("下一题",),
                    exclude=("答题后", "保存答案", "全部试题", "提交", "避免答题记录"),
                )
                if nxt is None:
                    nxt = self._find(
                        more, ("保存",), exclude=("提交", "暂存", "答题后", "全部试题")
                    )
            except Exception as exc:  # noqa: BLE001
                self.log.debug("底部按钮补 OCR 失败: %s", exc)
            if nxt is not None:
                for ratio in (0.15, 0.35, 0.0, -0.10):
                    y = nxt.rect.cy + int(ratio * nxt.rect.height)
                    self.mouse.click(nxt.rect.cx, y, "下一题/保存(底部补识别)")
                    time.sleep(float(getattr(self.cfg.practice, "question_wait", 0.8)))
                    after = self._current_question_number()
                    if before is not None and after is not None and after != before:
                        return True
                    self.log.warning(
                        "补识别后点击仍停在第 %s 题，重试落点", before
                    )
            warning = next((ln for ln in lines if "答题后" in ln.text and "下一题" in ln.text), None)
            if warning is not None:
                # “下一题”按钮就在红色提示语上方约 3.5% 屏高处；
                # 旧值 6% 会落在按钮上边缘甚至边框上，鼠标看着偏高。
                x = int(0.53 * frame.shape[1])
                y = warning.rect.y0 - int(0.035 * frame.shape[0])
                self.mouse.click(x, y, "下一题(位置推算)")
                time.sleep(float(getattr(self.cfg.practice, "question_wait", 0.8)))
                after = self._current_question_number()
                if before is not None and after is not None and after != before:
                    return True
        # 最后一层兜底：点击右侧答题卡的下一题号
        if before is not None and before < count:
            frame = self.screen.grab()
            numbers = self._answer_card_numbers(frame, count)
            target = numbers.get(before + 1)
            if target is not None and self._is_answer_card_rect(frame, target):
                self.mouse.click_rect(target, f"下一题(答题卡 {before + 1})")
                time.sleep(float(getattr(self.cfg.practice, "question_wait", 0.8)))
                after = self._current_question_number()
                if after is not None and after != before:
                    return True
        self.log.error("未能进入下一题（当前题号 %s，题目总数 %d）", before, count)
        return False

    def go_first(self, count: int) -> None:
        self.close_image_viewer()
        frame = self.screen.grab()
        lines = self._read(frame)
        prev = self._find(lines, ("上一题",))
        if prev is None:
            # 上一题按钮文字很浅时通常只识别到末尾的“题”
            nxt = self._find(lines, ("下一题",))
            if nxt is not None:
                candidates = [
                    ln
                    for ln in lines
                    if ln.text.strip().endswith("题")
                    and ln.rect.x1 <= nxt.rect.x0
                    and abs(ln.cy - nxt.rect.cy) <= 1.5 * nxt.rect.height
                ]
                if candidates:
                    prev = max(candidates, key=lambda ln: ln.rect.x1)
        if prev is None:
            self.log.info("未找到上一题按钮，按当前题目开始")
            return
        for _ in range(max(1, count) + 2):
            self.mouse.click_rect(prev.rect, "上一题")
            time.sleep(0.25)

    # ------------------------------------------------------------------ 保存
    def _confirm_stash(self) -> bool:
        frame = self.screen.grab()
        lines = self._read(frame)
        text = self._text(lines)
        kind = self._modal_kind(self._text(lines))
        if kind == "submit":
            cancel = self._find(lines, ("取消", "关闭"))
            if cancel is not None:
                self.mouse.click_rect(cancel.rect, "取消提交确认")
            self.log.error("检测到提交确认弹窗，已取消；本脚本绝不提交作业")
            return False
        if kind == "stash":
            confirm = self._find(lines, ("确定", "确认"), exact=True)
            if confirm is not None:
                self.mouse.click_rect(confirm.rect, "确定暂存")
                time.sleep(1.0)
                return True
        return True

    @staticmethod
    def _modal_kind(text: str) -> Optional[str]:
        """识别确认弹窗：stash / submit / None。"""
        modal_text = " ".join(
            ln.strip()
            for ln in (text or "").splitlines()
            if any(
                k in ln
                for k in (
                    "确定要",
                    "确认提交",
                    "暂存作业吗",
                    "提交本次",
                    "再次提交",
                    "无法再次提交",
                )
            )
        )
        if any(
            k in modal_text
            for k in ("确认提交", "提交本次", "提交作业吗", "再次提交", "无法再次提交")
        ) or ("提交" in modal_text and "暂存" not in modal_text):
            return "submit"
        if "暂存" in modal_text or ("确定要" in modal_text and "暂存" in text):
            return "stash"
        return None

    def save_draft(self) -> bool:
        self.close_image_viewer()
        frame = self.screen.grab()
        lines = self._read(frame)
        save = self._find_save(lines)
        if save is None:
            self.log.error("未找到‘暂存作业’，为避免误提交，不点击任何提交按钮")
            return False
        self.mouse.click_rect(save.rect, "暂存作业")
        time.sleep(0.9)
        return self._confirm_stash()

    @staticmethod
    def _looks_like_course(text: str) -> bool:
        return any(w in text for w in ("目录", "课程学习", "返回学堂", "学前必读", "教学大纲"))

    def _is_exam_page(self, text: str) -> bool:
        return (
            "单元测试" in text
            or "暂存作业" in text
            or "提交作业" in text
            or self._is_submitted_text(text)
        )

    def _active_window_title(self) -> str:
        try:
            from .platform_utils import get_active_window_title

            return get_active_window_title()
        except Exception:  # noqa: BLE001
            return ""

    def close_tab(self) -> bool:
        # 平台在“确定暂存”后会自动关闭考试标签。此时 OCR 可能只看到纯视频页，
        # 甚至还没读完“目录”；如果还按 Ctrl+W 就会把课程学习标签一起关掉。
        # 因此：只有明确看到考试页特征（标题或正文）才关，未知/课程页一律不动。
        title = self._active_window_title()
        exam_title = any(k in title for k in ("单元测试", "作业考试", "在线学堂", "考试"))
        course_title = any(k in title for k in ("课程学习", "返回学堂"))
        if course_title and not exam_title:
            return True
        text = self._text(self._read(self.screen.grab()))
        if not self._is_exam_page(text) and not exam_title:
            return True
        self.mouse.hotkey("ctrl", "w")
        time.sleep(1.0)
        text = self._text(self._read(self.screen.grab()))
        return self._looks_like_course(text) or not text.strip()

    # ------------------------------------------------------------------ 主流程
    def run(self, close_tab: bool = True) -> dict:
        if not self.wait_for_page():
            self.log.error("练习页在超时时间内仍未就绪（可能白屏/网络慢），不判定为完成")
            if close_tab:
                self.close_tab()
            return {"opened": False, "not_ready": True}
        self.close_image_viewer()
        frame = self.screen.grab()
        lines = self._read(frame)
        text = self._text(lines)
        chapter = self._chapter(text)
        if self._is_submitted_text(text):
            self.log.info("练习已提交/不可再次提交，跳过: %s", chapter or "")
            if close_tab:
                self.close_tab()
            return {"opened": True, "submitted": True, "chapter": chapter}
        if self._find_save(lines) is None:
            self.log.warning("没有‘暂存作业’，页面可能未加载完整；关闭标签后稍后重试")
            if close_tab:
                self.close_tab()
            return {"opened": False, "not_ready": True, "chapter": chapter}

        count = self._question_count(lines, text, frame)
        if count == int(getattr(self.cfg.practice, "default_questions", 10)) and not COUNT_RE.search(text):
            # 页面元信息可能比“暂存”按钮晚半秒出现；再读一次，避免 3 题被当成 10 题
            time.sleep(0.8)
            frame = self.screen.grab()
            lines = self._read(frame)
            text = self._text(lines)
            count = self._question_count(lines, text, frame)
        self.log.info("练习页面: %s，题目数 %d，准备开始（只暂存，不提交）", chapter or "?", count)
        stats = {"answered": 0, "skipped": 0, "failed": 0}
        # 不再点击右侧答题卡跳题：置顶 GUI/小条可能盖住答题卡，
        # 点击“第 2 题”会被 GUI 接住并把它唤到前台。
        # 统一用底部“下一题/保存”顺序前进；答题卡只在统计题数时读取。
        use_numbers = False
        self.log.info("练习导航：先定位第一个未完成题，再使用‘下一题’顺序前进")
        frame = self.screen.grab()
        first = self._first_unanswered(frame, count)
        current = self._current_question_number(frame)
        self.log.info("答题卡扫描: 当前第 %s 题，首个未完成第 %s 题", current, first)
        if current is not None and current != first:
            numbers = self._answer_card_numbers(frame, count)
            target = numbers.get(first)
            if target is not None and self._is_answer_card_rect(frame, target):
                self.mouse.click_rect(target, f"跳到第 {first} 题")
                time.sleep(float(getattr(self.cfg.practice, "question_wait", 0.8)))
                after = self._current_question_number()
                if after != first:
                    self.log.warning("答题卡跳题未成功（%s -> %s），从当前题继续", current, after)
        for index in range(max(0, first - 1), count):
            if self._stopped():
                self.log.warning("收到停止信号，停止练习")
                break
            try:
                result = self.answer_current()
            except Exception as exc:  # noqa: BLE001
                # 一道题失败不能拖垮整场练习；记录后继续下一题
                self.log.exception("第 %d 题处理异常，跳过该题继续: %s", index + 1, exc)
                result = "failed"
            stats[result] = stats.get(result, 0) + 1
            if index < count - 1:
                if not self.next_question(count, index):
                    message = (
                        f"第 {index + 1}/{count} 题后翻页失败，已保留已答内容并暂存；"
                        "将继续下一份练习。"
                    )
                    self.log.warning(message)
                    if self.notifier is not None:
                        try:
                            self.notifier.send(
                                "练习未完成（翻页失败）", message, "warning", force=True
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    break
                time.sleep(float(getattr(self.cfg.practice, "question_wait", 0.8)))
        saved = self.save_draft()
        if close_tab:
            self.close_tab()
        self.log.info("练习处理完成: %s，暂存=%s", stats, saved)
        return {
            "opened": True,
            "chapter": chapter,
            "count": count,
            "saved": saved,
            **stats,
        }


class PracticeRunner:
    """扫描课程目录里的“平时测试”，逐个打开新标签页处理。"""

    def __init__(
        self,
        ocr: OcrEngine,
        cfg: Config,
        mouse,
        screen,
        llm,
        state,
        navigator: PanelNavigator,
        logger: Any = None,
        stop_event=None,
        notifier=None,
        pause_event=None,
    ):
        self.ocr = ocr
        self.cfg = cfg
        self.mouse = mouse
        self.screen = screen
        self.llm = llm
        self.state = state
        self.navigator = navigator
        self.log = logger or log
        self.stop_event = stop_event
        self.notifier = notifier
        self.pause_event = pause_event
        self.scanner = PracticeScanner(ocr, cfg)

    def _stopped(self) -> bool:
        if self.pause_event is not None:
            while self.pause_event.is_set():
                if self.stop_event is not None and self.stop_event.is_set():
                    return True
                time.sleep(0.15)
        return bool(self.stop_event is not None and self.stop_event.is_set())

    def run(self, max_tests: int = 0) -> dict:
        self.navigator.scroll_to_top()
        done: set[str] = set()
        attempts: dict[str, int] = {}
        completed_skipped = 0
        processed = 0
        results: list[dict] = []
        while not self._stopped():
            frame = self.screen.grab()
            entries = self.scanner.scan(frame, self.navigator.layout)
            target = None
            for entry in entries:
                if entry.key in done or entry.chapter in done:
                    continue
                if self.state.is_practice_done(entry.key):
                    self.log.info("记录显示该平时测试已暂存/提交，跳过: %s", entry)
                    done.add(entry.key)
                    completed_skipped += 1
                    continue
                if entry.completed:
                    self.log.info("平时测试已完成（蓝勾），点击前跳过: %s", entry)
                    done.add(entry.key)
                    completed_skipped += 1
                    continue
                target = entry
                break
            if target is None:
                if self.navigator.scroll_down():
                    continue
                break
            if max_tests and processed >= max_tests:
                break
            self.log.info("打开练习: %s", target)
            self.mouse.click_rect(target.rect, "平时测试")
            time.sleep(float(getattr(self.cfg.practice, "open_wait", 2.0)))
            page = PracticePage(
                self.ocr,
                self.cfg,
                self.mouse,
                self.screen,
                self.llm,
                self.state,
                self.log.getChild("page"),
                self.stop_event,
                notifier=self.notifier,
                pause_event=self.pause_event,
            )
            try:
                result = page.run(close_tab=True)
            except Exception as exc:  # noqa: BLE001
                self.log.exception("练习处理失败: %s", exc)
                result = {"opened": False, "error": str(exc)}
                page.close_tab()
            if result.get("not_ready"):
                n = attempts.get(target.key, 0) + 1
                attempts[target.key] = n
                if n < 2 and not self._stopped():
                    self.log.warning("页面未就绪，第 %d 次重试: %s", n, target)
                    time.sleep(1.0)
                    continue
                self.log.error("页面加载失败，重试后仍跳过（不标记完成）: %s", target)
                done.add(target.key)
                if target.chapter:
                    done.add(target.chapter)
                results.append(result)
                continue
            done.add(target.key)
            if target.chapter:
                done.add(target.chapter)
            if result.get("chapter"):
                done.add(result["chapter"])
            if result.get("saved") or result.get("submitted"):
                self.state.mark_practice(target.key, target.chapter, target.title)
            results.append(result)
            processed += 1
            self.state.save()
            time.sleep(0.8)
        summary = {
            "tests": processed,
            "submitted": sum(1 for r in results if r.get("submitted")),
            "answered": sum(int(r.get("answered", 0)) for r in results),
            "skipped": sum(int(r.get("skipped", 0)) for r in results),
            "saved": sum(1 for r in results if r.get("saved")),
            "completed_skipped": completed_skipped,
            "results": results,
        }
        self.log.info("自动练习结束: %s", summary)
        return summary
