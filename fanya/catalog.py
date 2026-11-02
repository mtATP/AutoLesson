"""泛雅右侧目录扫描。

目录节点形如：
    1.4.1 中国武装力量性质、宗旨、使命
      1.3.1 国防体制
右侧状态图标：
- 绿色勾（icon_Completed）＝该知识点的所有任务点已完成；
- 橙色数字（catalog_points_yi）＝还有待完成任务点，数字是待完成数量。
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from autolesson.ocr import OcrEngine, sort_lines
from autolesson.types import OcrLine, Rect

from .layout import FanyaLayout
from .types import (
    STATUS_COMPLETED,
    STATUS_PENDING,
    STATUS_UNKNOWN,
    FanyaItem,
)

log = logging.getLogger("fanya.catalog")

# 1.4.1 / 1.3.2 / 2.1.1；允许全角点
ITEM_RE = re.compile(r"^\s*(\d+(?:[.．]\d+){1,3})\s*[、.．]?\s*(.+?)\s*$")
# 状态图标颜色（BGR -> HSV）
GREEN_LO = np.array([35, 70, 70], dtype=np.uint8)
GREEN_HI = np.array([95, 255, 255], dtype=np.uint8)
ORANGE_LO = np.array([5, 90, 90], dtype=np.uint8)
ORANGE_HI = np.array([30, 255, 255], dtype=np.uint8)


class FanyaCatalogScanner:
    def __init__(
        self,
        ocr: OcrEngine,
        cfg=None,
        min_item_level: int = 2,
        upscale: float = 1.5,
    ):
        self.ocr = ocr
        self.cfg = cfg
        self.min_item_level = int(min_item_level)
        self.upscale = float(upscale)

    def scan(self, frame: np.ndarray, layout: FanyaLayout) -> list[FanyaItem]:
        crop = layout.list_rect.crop(frame)
        if crop.size == 0:
            return []
        lines = self.ocr.read(
            crop,
            offset=(layout.list_rect.x0, layout.list_rect.y0),
            upscale=self.upscale,
        )
        lines = sort_lines(lines, row_tolerance=max(6, int(0.012 * layout.screen_w)))
        ai_rect = self._ai_tutor_rect(frame, layout)
        if ai_rect is not None:
            log.debug("检测到泛雅 AI 助教: %s，遮挡行不参与状态判断", ai_rect)
        return self.parse(lines, frame, layout, ai_rect=ai_rect)

    def parse(
        self,
        lines: list[OcrLine],
        frame: np.ndarray,
        layout: FanyaLayout,
        ai_rect: Optional[Rect] = None,
    ) -> list[FanyaItem]:
        panel = layout.list_rect
        # 按编号去重：同一编号在不同滚动位置可能被 OCR 成截断标题，
        # 不能按“编号|标题”当两个任务点，否则会重复点击/点错。
        items: dict[str, FanyaItem] = {}
        ai_skipped = 0
        count_candidates: list[tuple[int, int]] = []
        for ln in lines:
            text = ln.text.strip()
            if re.fullmatch(r"[012]", text) and ln.rect.x0 > layout.panel.x1 - 140:
                count_candidates.append((ln.rect.cy, int(text)))
        for line in lines:
            text = (line.text or "").strip()
            if not text:
                continue
            match = ITEM_RE.match(text)
            if not match:
                continue
            raw_index = match.group(1).replace("．", ".")
            title = match.group(2).strip(" .·•…")
            level = raw_index.count(".")
            if level < self.min_item_level:
                continue
            if not title:
                continue
            if DURATION_RE.fullmatch(title):
                continue
            rect = line.rect
            if rect.cy < panel.y0 + 4 or rect.cy > panel.y1 - 4:
                continue
            if ai_rect is not None and self._status_overlaps_ai(layout, rect.cy, ai_rect):
                # AI 助教的橙色星星会在状态列被读成“待完成”，绿色勾也会
                # 被挡住。这一行先不参与计划；滚出 AI 区域后自然会再读到。
                log.debug("忽略被 AI 助教遮挡的任务点: %s %s", raw_index, title)
                ai_skipped += 1
                continue
            status = self._status_at(frame, layout, rect.cy)
            font_h = int(np.clip(rect.height, 10, 40))
            pending_count: Optional[int] = None
            if status != STATUS_COMPLETED:
                nearest = min(
                    count_candidates,
                    key=lambda item: abs(item[0] - rect.cy),
                    default=None,
                )
                if nearest is not None and abs(nearest[0] - rect.cy) <= max(12, int(0.9 * font_h)):
                    pending_count = nearest[1]
                if pending_count is None:
                    pending_count = self._read_status_count(
                        frame, layout, rect.cy, font_h
                    )
                if pending_count == 0:
                    status = STATUS_COMPLETED
            current = self._is_current_row(frame, layout, rect)
            row_top = max(panel.y0, rect.cy - int(0.018 * layout.screen_h))
            row_bottom = min(panel.y1, rect.cy + int(0.018 * layout.screen_h))
            item = FanyaItem(
                index=raw_index,
                title=title,
                level=level,
                status=status,
                row=Rect(layout.panel.x0, row_top, layout.panel.x1, row_bottom),
                title_rect=rect,
                current=current,
                pending_count=pending_count,
            )
            existing = items.get(raw_index)
            if existing is None or prefer_item(item, existing):
                items[raw_index] = item
        result = sorted(items.values(), key=lambda x: _index_order(x.index))
        if ai_skipped:
            log.info("AI 助教遮挡 %d 个任务点，已跳过（滚出遮挡后会重新读取）", ai_skipped)
        log.debug("泛雅目录解析到 %d 个任务点", len(result))
        return result

    def _read_status_count(
        self, frame: np.ndarray, layout: FanyaLayout, cy: int, font_h: int
    ) -> Optional[int]:
        if self.ocr is None:
            return None
        zone = Rect(
            max(layout.panel.x0, layout.screen_w - 150),
            max(0, cy - 30),
            layout.screen_w - 4,
            min(frame.shape[0], cy + 30),
        )
        try:
            lines = self.ocr.read(zone.crop(frame), offset=(zone.x0, zone.y0), upscale=2.5)
        except Exception:  # noqa: BLE001
            return None
        for line in lines:
            m = re.search(r"[012]", line.text)
            if m:
                return int(m.group(0))
        return None

    @staticmethod
    def _status_zone(layout: FanyaLayout, cy: int) -> Rect:
        w = layout.screen_w
        return Rect(
            max(layout.panel.x0, w - 92),
            max(0, cy - 26),
            w - 6,
            cy + 26,
        )

    @classmethod
    def _status_overlaps_ai(
        cls, layout: FanyaLayout, cy: int, ai_rect: Rect
    ) -> bool:
        zone = cls._status_zone(layout, cy)
        return not (
            zone.x1 <= ai_rect.x0
            or zone.x0 >= ai_rect.x1
            or zone.y1 <= ai_rect.y0
            or zone.y0 >= ai_rect.y1
        )

    @staticmethod
    def _ai_tutor_rect(frame: np.ndarray, layout: FanyaLayout) -> Optional[Rect]:
        """定位右下角 AI 助教（常态紫色徽章 / 悬停黄色星星）。

        不清空整个底部：只把状态列与助教图形重叠的行排除，助教下方的
        条目仍可在滚出遮挡后正常读取。
        """
        panel = layout.panel
        x0 = panel.x0 + int(0.45 * panel.width)
        y0 = panel.y0 + int(0.45 * panel.height)
        crop = frame[y0 : panel.y1, x0 : panel.x1]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 常态徽章是紫色底白字；悬停星球是黄色。
        purple = cv2.inRange(
            hsv, np.array([120, 70, 70], np.uint8), np.array([165, 255, 255], np.uint8)
        )
        yellow = cv2.inRange(
            hsv, np.array([15, 120, 150], np.uint8), np.array([40, 255, 255], np.uint8)
        )
        found: dict[str, tuple[int, tuple[int, int, int, int]]] = {}
        for kind, mask, min_area in (("purple", purple, 260), ("yellow", yellow, 900)):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                area = int(cv2.contourArea(contour))
                if area < min_area:
                    continue
                bx, by, bw, bh = cv2.boundingRect(contour)
                if bw < 18 or bh < 12:
                    continue
                if kind not in found or area > found[kind][0]:
                    found[kind] = (area, (bx, by, bw, bh))
        # 常态的紫色徽章最可靠；悬停星球没有徽章时再用黄色星球。
        if "purple" in found:
            kind = "purple"
        elif "yellow" in found:
            kind = "yellow"
        else:
            return None
        _area, (bx, by, bw, bh) = found[kind]
        if kind == "purple":
            # 徽章只是头像底部；AI 头像和对话气泡都在其上方。
            top = by - max(int(2.2 * bh), int(0.05 * panel.height))
            left = bx - 8
        else:
            top = by - 8
            left = bx - 8
        return Rect(x0 + max(0, left), y0 + max(0, top), panel.x1, y0 + by + bh + 6)

    @staticmethod
    def _status_at(frame: np.ndarray, layout: FanyaLayout, cy: int) -> str:
        """只右侧状态列找绿色勾/橙色数字，避免把标题里的蓝字当状态。"""
        w = layout.screen_w
        x0 = max(layout.panel.x0, w - 92)
        x1 = w - 6
        y0 = max(0, cy - 26)
        y1 = min(frame.shape[0], cy + 26)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return STATUS_UNKNOWN
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
        orange = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
        green_px = int(cv2.countNonZero(green))
        orange_px = int(cv2.countNonZero(orange))
        if green_px >= 12 and green_px >= orange_px:
            return STATUS_COMPLETED
        if orange_px >= 12:
            return STATUS_PENDING
        return STATUS_UNKNOWN

    @staticmethod
    def _is_current_row(frame: np.ndarray, layout: FanyaLayout, rect: Rect) -> bool:
        """当前目录行是浅蓝渐变背景；在标题左侧取一条背景色判断。"""
        x = max(layout.panel.x0 + 2, rect.x0 - 24)
        y = rect.cy
        crop = frame[max(0, y - 4) : y + 5, x : x + 10]
        if crop.size == 0:
            return False
        b, g, r = (float(v) for v in crop.mean(axis=(0, 1)))
        return b > 235 and r > 220 and b - r >= 8


def prefer_item(new: FanyaItem, old: FanyaItem) -> bool:
    """同一编号的两次 OCR：绿勾最可靠，绝不被后续 pending 覆盖。"""
    if old.status == STATUS_COMPLETED and new.status != STATUS_COMPLETED:
        return False
    if new.status == STATUS_COMPLETED and old.status != STATUS_COMPLETED:
        return True
    if new.status != STATUS_UNKNOWN and old.status == STATUS_UNKNOWN:
        return True
    if new.pending_count is not None and old.pending_count is None:
        return True
    if len(new.title) > len(old.title) + 2:
        return True
    return False


def _index_order(index: str) -> tuple:
    parts = []
    for part in index.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


DURATION_RE = re.compile(r"^\d{1,3}\s*[:：]\s*\d{1,2}\s*[:：]\s*\d{1,2}$")
