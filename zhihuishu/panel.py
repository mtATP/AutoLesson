from __future__ import annotations

import logging
import re
import time
from typing import Optional

import numpy as np

from autolesson import vision
from autolesson.config import Config
from .layout import Layout
from autolesson.ocr import OcrEngine, sort_lines
from autolesson.types import (
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    STATUS_NOT_STARTED,
    Lesson,
    OcrLine,
    Rect,
    normalize_title,
)

log = logging.getLogger("autolesson.panel")

# 兼容真实站点 hh:mm:ss 与复刻页 mm:ss
DURATION_RE = re.compile(r"(?:(\d{1,3})\s*:\s*)?(\d{1,2})\s*:\s*(\d{2})")
# OCR 偶尔把时长读成 00:12.70 / 00：12：30，不能并进标题，否则标题行变高会被丢掉
CLOCK_NOISE_RE = re.compile(
    r"^[●•·\s]*(?:\d{1,3}\s*[:：]\s*)?\d{1,2}\s*[:：]\s*\d{1,2}\s*$"
)
INDEX_RE = re.compile(r"^\s*(\d{1,2}(?:\s*[.．]\s*\d{1,2}){1,3})\s*$")
PERCENT_RE = re.compile(r"(\d{1,3})\s*%")
PERCENT_REV_RE = re.compile(r"%\s*(\d{1,3})\s*%")
CHAPTER_RE = re.compile(r"第\s*[一二三四五六七八九十百零〇\d]+\s*章")
COURSE_NO_RE = re.compile(r"第\s*\d+\s*课")
TEST_WORDS = ("平时测试", "平时测验", "章节测试")


def parse_duration(text: str) -> Optional[int]:
    m = DURATION_RE.search(text or "")
    if not m:
        return None
    hour = int(m.group(1)) if m.group(1) is not None else 0
    minute, sec = int(m.group(2)), int(m.group(3))
    if minute >= 60 or sec >= 60:
        return None
    return hour * 3600 + minute * 60 + sec


def parse_percent(text: str) -> Optional[int]:
    m = PERCENT_RE.search(text or "")
    if m:
        return int(m.group(1))
    m = PERCENT_REV_RE.search(text or "")
    if m:
        return int(m.group(1))
    return None


def _join_title(parts: list[str]) -> str:
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and out[-1].isascii() and out[-1].isalnum() and part[0].isascii() and part[0].isalnum():
            out += " " + part
        else:
            out += part
    return out.strip(" .·•…")


class PanelScanner:
    """解析右侧课程目录。只依赖截图 + OCR。"""

    def __init__(self, ocr: OcrEngine, cfg: Config):
        self.ocr = ocr
        self.cfg = cfg

    # ------------------------------------------------------------------ 扫描
    def scan(
        self, frame: np.ndarray, layout: Layout, upscale: Optional[float] = None
    ) -> list[Lesson]:
        panel = layout.panel
        if panel.width < 80 or panel.height < 80:
            return []
        crop = panel.crop(frame)
        if upscale is None:
            upscale = float(self.cfg.panel.ocr_upscale or 0)
        if upscale <= 0:
            upscale = 1.0
        lines = self.ocr.read(crop, offset=(panel.x0, panel.y0), upscale=upscale)
        lines = sort_lines(lines, row_tolerance=max(6, int(0.012 * layout.screen_w)))
        return self.parse(lines, frame, layout)

    @staticmethod
    def _is_system_tray_text(rect: Rect, layout: Layout) -> bool:
        """OCR 行是否落在屏幕右下角的系统托盘（任务栏时钟/日期）。

        目录裁剪区域的下边界就是屏幕底部，Windows 任务栏会一起进图；
        托盘里的 ``15:35`` 能被 ``parse_duration`` 读成时长，必须排除。
        """
        panel = layout.panel
        tray_top = layout.screen_h - max(48, int(0.045 * layout.screen_h))
        tray_left = panel.x1 - max(100, int(0.30 * panel.width))
        return rect.y0 >= tray_top and rect.x0 >= tray_left

    def parse(self, lines: list[OcrLine], frame: np.ndarray, layout: Layout) -> list[Lesson]:
        panel = layout.panel
        duration_lines: list[tuple[OcrLine, int]] = []
        for line in lines:
            seconds = parse_duration(line.text)
            if seconds is None or seconds <= 0:
                continue
            duration_lines.append((line, seconds))
        if not duration_lines:
            return []

        lessons: dict[str, Lesson] = {}
        for duration_line, seconds in duration_lines:
            rect = duration_line.rect
            if rect.y0 < panel.y0 + 2 or rect.y1 > min(layout.screen_h, panel.y1) - 2:
                continue
            if self._is_system_tray_text(rect, layout) or (
                rect.cy > layout.screen_h - max(48, int(0.045 * layout.screen_h))
            ):
                # 目录面板一直裁到屏幕底部，会把 Windows 任务栏时钟（如 15:35）
                # 读成“最后一个视频”的时长。视频时长在标题下方左对齐，
                # 不会出现在屏幕右下角的系统托盘区域。
                log.debug("忽略任务栏区域的时间文本: %r @ %s", duration_line.text, rect)
                continue
            font_h = int(np.clip(rect.height, 10, 40))
            row_top = int(rect.y0 - float(self.cfg.panel.row_title_gap) * font_h)
            row_bottom = int(rect.y1 + 0.25 * font_h)
            row_rect = Rect(panel.x0, row_top, panel.x1, row_bottom)
            # 状态图标/百分比与标题-时长这条基线对齐，避免上一行的状态串到下一行
            anchor_y = int(rect.cy - 0.45 * font_h)

            title_parts: list[OcrLine] = []
            index: Optional[str] = None
            index_cy = 10**9
            index_line_cy: Optional[int] = None
            percent: Optional[int] = None
            for line in lines:
                if line is duration_line:
                    continue
                if line.text == duration_line.text and line.rect == duration_line.rect:
                    continue
                cy = line.cy
                in_row = row_top <= cy <= rect.y0 + 0.7 * font_h
                # 索引（1.1 / 2.3）
                m_index = INDEX_RE.match(line.text)
                if (
                    m_index
                    and line.rect.x1 < rect.x0 - 0.4 * font_h
                    and row_top - 0.5 * font_h <= cy <= rect.y1
                    # 编号要和本行时长垂直相邻；否则只露出标题的半截视频
                    # 会把上一行的编号抢过来，造成面包屑永恒不匹配。
                    and abs(cy - rect.cy) <= 1.6 * font_h
                ):
                    if abs(cy - rect.cy) < index_cy:
                        raw_index = re.sub(r"\s+", "", m_index.group(1)).replace("．", ".")
                        index = raw_index
                        index_cy = abs(cy - rect.cy)
                        index_line_cy = cy
                    continue
                if not in_row:
                    continue
                # 章标题字号明显更大，避免把“矩阵及其运算”之类合并进小节标题
                if line.rect.height > 1.6 * font_h:
                    continue
                if line.rect.cx > panel.x1 - int(self.cfg.panel.right_icon_width):
                    if abs(line.cy - anchor_y) <= 1.1 * font_h:
                        pct = parse_percent(line.text)
                        if pct is not None:
                            percent = pct
                    continue
                text = line.text.strip()
                if not text:
                    continue
                if DURATION_RE.fullmatch(text) or CLOCK_NOISE_RE.fullmatch(text):
                    continue
                if INDEX_RE.match(text):
                    continue
                # 真站章标题是“第一章 行列式”；复刻页视频标题是“第1章第1课”，
                # 后者带“第N课”，不能当章标题丢掉。
                if text.startswith("第") and CHAPTER_RE.search(text) and not COURSE_NO_RE.search(text):
                    continue
                if CHAPTER_RE.search(text) and line.rect.height > 1.5 * font_h:
                    continue
                if any(word in text for word in TEST_WORDS):
                    continue
                if set(text) <= set("·•.…-—_ "):
                    continue
                title_parts.append(line)

            if not title_parts:
                continue
            if index_line_cy is not None:
                # 标题必须和本行编号垂直对齐。章标题（如“函数”）在编号上方
                # 较远，页边界处会被错配成下一条视频的时长，必须排除。
                aligned = [
                    ln for ln in title_parts if abs(ln.cy - index_line_cy) <= 1.7 * font_h
                ]
                if not aligned:
                    continue
                title_parts = aligned
            title_parts.sort(key=lambda ln: ln.cx)
            title = _join_title([ln.text for ln in title_parts])
            if not index:
                # OCR 有时把编号和标题合成一行，例如
                # “3.3.1 数据与数据类型（上）”；不能把编号当标题前缀删掉。
                m_lead = re.match(
                    r"^\s*(\d{1,2}(?:\s*[.．]\s*\d{1,2}){1,3})\s*[、.．]?\s*(.+)$",
                    title,
                )
                if m_lead:
                    index = re.sub(r"\s+", "", m_lead.group(1)).replace("．", ".")
                    title = m_lead.group(2).strip()
            # 没读到编号的半行只能等下一次重叠滚动后重新识别，不能参与
            # 待学队列。否则任务栏时钟会和页尾任意标题拼出假视频。
            if not index:
                log.debug("忽略缺少编号的目录残片: %r", title)
                continue
            if len(normalize_title(title)) < int(self.cfg.panel.min_title_chars):
                continue
            if any(word in title for word in TEST_WORDS):
                continue
            if DURATION_RE.fullmatch(title.strip()) or CLOCK_NOISE_RE.fullmatch(title.strip()):
                continue
            # 时长和标题位于同一文字列。跨列的托盘时钟即使靠近编号
            # 也不能用来组成视频；检查区间重叠而非固定屏幕坐标。
            if not any(
                part.rect.x0 - font_h <= rect.x0 <= part.rect.x1 + font_h
                for part in title_parts
            ):
                continue
            title_rect = title_parts[0].rect
            for part in title_parts[1:]:
                title_rect = title_rect.union(part.rect)
            if title_rect.height > 1.6 * font_h:
                # 误把章标题合并进来
                continue

            zone = Rect(
                panel.x1 - int(self.cfg.panel.right_icon_width),
                row_top,
                panel.x1,
                row_bottom,
            )
            status, _icon_rect = vision.find_status_icon(
                frame, zone, row_top, row_bottom, font_h, anchor_y=anchor_y
            )
            if status == STATUS_COMPLETED:
                pass
            elif status == STATUS_IN_PROGRESS:
                pass
            elif percent is not None and percent < 100:
                status = STATUS_IN_PROGRESS
            else:
                status = STATUS_NOT_STARTED
            # 若图标识别成实心圆但仍显示未满进度，优先相信百分比
            if status == STATUS_COMPLETED and percent is not None and percent < 100:
                status = STATUS_IN_PROGRESS

            lesson = Lesson(
                index=index,
                title=title,
                duration=seconds,
                status=status,
                title_rect=title_rect,
                row=row_rect,
                percent=percent,
            )
            existing = lessons.get(lesson.key)
            if existing is None or (lesson.percent or -1) > (existing.percent or -1):
                lessons[lesson.key] = lesson

        result = sorted(lessons.values(), key=lambda l: (l.title_rect.cy, l.title_rect.cx))
        log.debug("目录解析到 %d 个视频", len(result))
        return result


class PanelNavigator:
    """目录滚动与全量扫描（回退到拼接重构前的逐页机制）。"""

    SIG_THRESHOLD = 40

    def __init__(
        self,
        scanner: PanelScanner,
        ocr: OcrEngine,
        cfg: Config,
        mouse,
        screen,
        layout: Layout,
        pause=None,
    ):
        self.scanner = scanner
        self.ocr = ocr
        self.cfg = cfg
        self.mouse = mouse
        self.screen = screen
        self.layout = layout
        self._pause = pause
        self._last_lessons: list[Lesson] = []
        self.last_scan_complete = False
        self._view_step = 0

    @property
    def center(self) -> tuple[int, int]:
        panel = self.layout.panel
        return panel.cx, min(panel.y1 - 30, panel.y0 + int(0.6 * panel.height))

    def _scroll_step(self) -> int:
        """一页滚动的目标像素数。

        优先级：scroll_step_px > scroll_step_ratio > 旧版 scroll_step × 90。
        比例基于目录可视高度，换分辨率/DPI 后不用重调像素值。
        """
        panel_cfg = self.cfg.panel
        px = int(getattr(panel_cfg, "scroll_step_px", 0) or 0)
        if px:
            step = min(2000, max(60, abs(px)))
            log.debug("目录滚动步长: %dpx（scroll_step_px）", step)
            return step
        ratio = float(getattr(panel_cfg, "scroll_step_ratio", 0) or 0)
        if ratio:
            step = min(2000, max(60, int(self.layout.panel.height * abs(ratio))))
            log.debug("目录滚动步长: %dpx（scroll_step_ratio=%.2f）", step, abs(ratio))
            return step
        step = min(600, max(300, abs(int(panel_cfg.scroll_step)) * 90))
        log.debug("目录滚动步长: %dpx（旧版 scroll_step）", step)
        return step

    def _scroll_wait(self) -> float:
        return min(0.35, float(self.cfg.panel.scroll_wait))

    def _wheel_scroll(self, pixels: int, direction: int) -> None:
        """连续发送小步滚轮事件，累计约 pixels 像素。

        实测 Windows/Chrome 下 pyautogui 的一次滚轮事件约只走几十像素，
        单个 scroll(-20) 远小于一行；拆成 60 像素的小事件才能稳定走一屏。
        """
        remaining = abs(int(pixels))
        x, y = self.center
        while remaining > 0:
            chunk = min(60, remaining)
            self.mouse.scroll_at(x, y, int(direction) * chunk)
            remaining -= chunk
            time.sleep(0.03)

    def _max_pages(self) -> int:
        return max(12, min(40, int(self.cfg.panel.max_scrolls)))

    def scan(
        self, frame: Optional[np.ndarray] = None, upscale: Optional[float] = None
    ) -> list[Lesson]:
        frame = self.screen.grab() if frame is None else frame
        if upscale is None:
            self._last_lessons = self.scanner.scan(frame, self.layout)
        else:
            self._last_lessons = self.scanner.scan(frame, self.layout, upscale=upscale)
        return self._last_lessons

    def _signature_of(self, img: np.ndarray) -> bytes:
        margin = max(int(self.cfg.panel.right_icon_width), 110)
        h, w = img.shape[:2]
        keep_w = max(40, min(w - margin, int(0.75 * w)))
        keep_h = max(40, int(0.92 * h))
        return vision.image_signature(img[:keep_h, :keep_w])

    def _current_signature(self) -> bytes:
        """只截目录文字区域算指纹，排除右侧状态图标、滚动条和右下角 AI 助教动画。"""
        margin = max(int(self.cfg.panel.right_icon_width), 110)
        try:
            return self._signature_of(self.screen.grab_rect(self.layout.panel))
        except Exception:  # noqa: BLE001
            return vision.panel_scroll_signature(self.screen.grab(), self.layout.panel, margin)

    def _wait_settle(self, before: bytes) -> bytes:
        max_wait = max(0.15, self._scroll_wait())
        deadline = time.monotonic() + max_wait
        last = before
        stable = 0
        while time.monotonic() < deadline:
            time.sleep(0.06)
            sig = self._current_signature()
            if vision.signature_distance(sig, last) <= 8:
                stable += 1
                if stable >= 2:
                    return sig
            else:
                stable = 0
                last = sig
        return last

    def _drag_scrollbar_top(self) -> None:
        panel = self.layout.panel
        x = panel.x1 - 6
        y0 = panel.y0 + int(0.55 * panel.height)
        y1 = panel.y0 + 8
        try:
            self.mouse.drag(x, y0, x, y1, "目录滚动到顶部")
            time.sleep(0.25)
        except Exception as exc:  # noqa: BLE001
            log.debug("拖动目录滚动条失败: %s", exc)

    def _drag_scrollbar_down(self) -> None:
        panel = self.layout.panel
        x = panel.x1 - 6
        y0 = panel.y0 + int(0.35 * panel.height)
        y1 = panel.y0 + int(0.85 * panel.height)
        try:
            self.mouse.drag(x, y0, x, y1, "目录向下拖动")
            time.sleep(0.25)
        except Exception as exc:  # noqa: BLE001
            log.debug("向下拖动目录滚动条失败: %s", exc)

    def scroll_to_top(self) -> None:
        """用滚轮回到顶部；滚轮点固定在列表正文，避免拖错滚动条。"""
        step = max(800, int(1.2 * self.layout.panel.height))
        sig = self._current_signature()
        unchanged = 0
        for _ in range(12):
            self._wheel_scroll(step, +1)
            time.sleep(0.40)
            new_sig = self._wait_settle(sig)
            if vision.signature_distance(sig, new_sig) <= self.SIG_THRESHOLD:
                unchanged += 1
                if unchanged >= 2:
                    break
            else:
                unchanged = 0
                sig = new_sig
        log.debug("目录已回到顶部")

    def scroll_down(self, step: Optional[int] = None) -> bool:
        """滚动约半屏，只比较图像指纹；OCR 留给 scan_all 每页做一次。"""
        before = self._current_signature()
        self._wheel_scroll(step or self._scroll_step(), -1)
        time.sleep(0.35)
        after = self._wait_settle(before)
        return vision.signature_distance(before, after) > self.SIG_THRESHOLD

    @staticmethod
    def _merge_lesson(seen: dict[str, Lesson], lesson: Lesson) -> None:
        # 同一视频在不同页可能被 OCR 截断成不同标题；按编号合并，
        # 否则 76 个视频会混进重复项，真正的 5.1/6.1 反而被挤掉。
        key = lesson.index or lesson.key
        existing = seen.get(key)
        if existing is None:
            seen[key] = lesson
            return
        if lesson.completed and not existing.completed:
            seen[key] = lesson
            return
        if existing.completed and not lesson.completed:
            return
        if len(normalize_title(lesson.title)) > len(normalize_title(existing.title)):
            seen[key] = lesson

    def scan_all(self) -> list[Lesson]:
        seen: dict[str, Lesson] = {}
        self.last_scan_complete = False
        self.scroll_to_top()
        unchanged = 0
        for page in range(self._max_pages()):
            if self._pause is not None:
                self._pause()
            lessons = self.scan(self.screen.grab())
            for lesson in lessons:
                if lesson.index:
                    self._merge_lesson(seen, lesson)
            log.info("目录扫描第 %d 页: 累计 %d 个视频", page + 1, len(seen))
            if page > 0 and not lessons:
                time.sleep(0.8)
            if not self.scroll_down():
                unchanged += 1
                if unchanged >= 2:
                    self.last_scan_complete = True
                    break
            else:
                unchanged = 0
        if self.last_scan_complete:
            log.info("目录全量扫描完成: %d 个视频", len(seen))
        else:
            log.error("目录达到 %d 页上限仍未到底，累计 %d 个视频",
                      self._max_pages(), len(seen))
        self.scroll_to_top()
        return list(seen.values())

    @staticmethod
    def find(lessons: list[Lesson], target: Lesson) -> Optional[Lesson]:
        # 先按编号找。5.1/6.1 都叫“问题的引入”，按标题找会点回第一条。
        if target.index:
            for lesson in lessons:
                if lesson.index == target.index:
                    return lesson
            return None
        # 运行时视频必须有编号；不再用无编号标题进行模糊定位。
        return None

    @staticmethod
    def incomplete(lessons: list[Lesson]) -> list[Lesson]:
        return [l for l in lessons if l.status != STATUS_COMPLETED]
