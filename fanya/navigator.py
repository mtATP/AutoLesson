"""泛雅右侧目录的滚动、全量扫描和条目定位。"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from autolesson import vision

from .catalog import FanyaCatalogScanner, prefer_item
from .layout import FanyaLayout
from .types import FanyaItem

log = logging.getLogger("fanya.navigator")


class FanyaCatalogNavigator:
    SIG_THRESHOLD = 40

    def __init__(
        self,
        scanner: FanyaCatalogScanner,
        screen,
        mouse,
        layout: FanyaLayout,
        cfg=None,
    ):
        self.scanner = scanner
        self.screen = screen
        self.mouse = mouse
        self.layout = layout
        self.cfg = cfg
        self.last_scan_complete = False

    @property
    def _center(self) -> tuple[int, int]:
        panel = self.layout.panel
        return panel.cx, panel.y0 + int(0.55 * panel.height)

    def _signature(self) -> bytes:
        """右侧状态图标和 AI 助教悬浮窗会动，滚动指纹只看左侧文字区。"""
        try:
            crop = self.layout.list_rect.crop(self.screen.grab())
            keep = max(80, crop.shape[1] - 140)
            return vision.image_signature(crop[:, :keep])
        except Exception:  # noqa: BLE001
            return b""

    def _wheel(self, pixels: int, direction: int) -> None:
        remaining = abs(int(pixels))
        x, y = self._center
        while remaining > 0:
            chunk = min(60, remaining)
            self.mouse.scroll_at(x, y, int(direction) * chunk)
            remaining -= chunk
            time.sleep(0.03)

    def scroll_to_top(self) -> None:
        # 不能只看到两次指纹相同就停：Chrome 平滑滚动/惯性会让中途两帧
        # 看起来一样，实测曾把顶部前 1.1/1.2 章整段漏掉。
        step = max(1200, int(1.4 * self.layout.list_rect.height))
        last = self._signature()
        unchanged = 0
        for _ in range(20):
            self._wheel(step, +1)
            time.sleep(0.55)
            first = self._signature()
            time.sleep(0.20)
            second = self._signature()
            if (
                vision.signature_distance(first, second) <= 8
                and vision.signature_distance(second, last) <= self.SIG_THRESHOLD
            ):
                unchanged += 1
                if unchanged >= 3:
                    break
            else:
                unchanged = 0
            last = second
        time.sleep(0.35)
        log.debug("泛雅目录已回到顶部")

    def _scroll_pixels(self) -> int:
        """泛雅目录每页滚动像素；px 优先，其次按可视高度比例。"""
        h = self.layout.list_rect.height
        fcfg = getattr(self.cfg, "fanya", None) if self.cfg is not None else None
        px = int(getattr(fcfg, "scroll_step_px", 0) or 0) if fcfg is not None else 0
        if px:
            step = min(2000, max(60, abs(px)))
            log.debug("泛雅目录滚动步长: %dpx（fanya.scroll_step_px）", step)
            return step
        ratio = float(getattr(fcfg, "scroll_step_ratio", 0.55) or 0) if fcfg is not None else 0.55
        if ratio <= 0:
            ratio = 0.55
        step = min(2000, max(60, int(h * ratio)))
        log.debug("泛雅目录滚动步长: %dpx（fanya.scroll_step_ratio=%.2f）", step, ratio)
        return step

    def scroll_down(self, pixels: Optional[int] = None) -> bool:
        step = int(pixels or self._scroll_pixels())
        before = self._signature()
        self._wheel(step, -1)
        time.sleep(0.4)
        after = self._signature()
        return vision.signature_distance(before, after) > self.SIG_THRESHOLD

    def scan_visible(self) -> list[FanyaItem]:
        return self.scanner.scan(self.screen.grab(), self.layout)

    def scan_all(self) -> list[FanyaItem]:
        self.last_scan_complete = False
        self.scroll_to_top()
        seen: dict[str, FanyaItem] = {}
        unchanged = 0
        for page in range(40):
            items = self.scan_visible()
            for item in items:
                old = seen.get(item.index)
                if old is None or prefer_item(item, old):
                    seen[item.index] = item
            if not self.scroll_down():
                unchanged += 1
                if unchanged >= 2:
                    self.last_scan_complete = True
                    break
            else:
                unchanged = 0
            log.info("泛雅目录扫描第 %d 页: 累计 %d 个任务点", page + 1, len(seen))
        self.scroll_to_top()
        result = sorted(seen.values(), key=lambda x: _order(x.index))
        log.info("泛雅目录扫描完成: %d 个任务点", len(result))
        return result

    def seek_forward(self, index: str) -> Optional[FanyaItem]:
        """从当前位置向下找目录条目；找不到再回顶找一遍。"""
        for _ in range(2):
            for _page in range(40):
                items = self.scan_visible()
                for item in items:
                    if item.index == index:
                        return item
                if not self.scroll_down():
                    break
            self.scroll_to_top()
        return None

    def seek_from_top(self, index: str) -> Optional[FanyaItem]:
        self.scroll_to_top()
        for _ in range(40):
            items = self.scan_visible()
            for item in items:
                if item.index == index:
                    return item
            if not self.scroll_down():
                break
        return None


def _order(index: str) -> tuple:
    parts = []
    for part in index.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)
