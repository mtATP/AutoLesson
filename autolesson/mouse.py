from __future__ import annotations

import logging
import random
import time
from typing import Optional

from .config import Config
from .platform_utils import enable_dpi_awareness
from .types import Rect, UserAbort


class InputUnavailable(RuntimeError):
    pass


class Mouse:
    """pyautogui 鼠标封装：随机抖动、限速、安全角中止。"""

    def __init__(self, cfg: Config, logger: Optional[logging.Logger] = None, dry_run: bool = False):
        self.cfg = cfg
        self.log = logger or logging.getLogger("autolesson.mouse")
        self.dry_run = bool(dry_run)
        self._gui = None
        self._last_action = 0.0
        self._origin = (0, 0)
        if not self.dry_run:
            enable_dpi_awareness()
            try:
                import pyautogui  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise InputUnavailable(
                    "无法加载 pyautogui，鼠标控制不可用。Windows 请确认已安装 requirements.txt；"
                    "Linux 还需安装 python3-xlib 与 scrot。"
                ) from exc
            pyautogui.FAILSAFE = bool(cfg.safety.fail_safe_corner)
            # 滚动阶段会连续调用，0.05 足够让事件落地，又不拖慢扫描
            pyautogui.PAUSE = 0.05
            self._gui = pyautogui

    def close(self) -> None:
        self._gui = None

    def set_origin(self, x: int, y: int) -> None:
        """截屏坐标 -> 全局鼠标坐标的偏移（多显示器时左侧/上侧可能有偏移）。"""
        self._origin = (int(x), int(y))
        if self._origin != (0, 0):
            self.log.info("鼠标坐标偏移: (%d,%d)", self._origin[0], self._origin[1])

    def _global(self, x: int, y: int) -> tuple[int, int]:
        return int(x) + self._origin[0], int(y) + self._origin[1]

    # ------------------------------------------------------------------ 基础
    def _throttle(self) -> None:
        interval = float(self.cfg.safety.min_action_interval)
        wait = interval - (time.monotonic() - self._last_action)
        if wait > 0:
            time.sleep(wait)
        self._last_action = time.monotonic()

    def _safe(self, fn, *args, **kwargs):
        if self._gui is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            if "FailSafe" in name:
                raise UserAbort("鼠标移动到屏幕左上角，已触发安全中止") from exc
            raise

    def position(self) -> tuple[int, int]:
        if self._gui is None:
            return (0, 0)
        point = self._safe(self._gui.position)
        return int(point.x), int(point.y)

    def move_to(self, x: int, y: int, duration: float = 0.25) -> None:
        self._throttle()
        gx, gy = self._global(x, y)
        if self.dry_run or self._gui is None:
            self.log.debug("[dry-run] 移动鼠标到 (%d,%d)", gx, gy)
            return
        self._safe(self._gui.moveTo, gx, gy, duration=float(duration))

    def wiggle(self, x: int, y: int) -> None:
        """轻微移动鼠标以唤出播放器控件，不点击。"""
        jitter_x = x + random.randint(-3, 3)
        jitter_y = y + random.randint(-3, 3)
        self.move_to(jitter_x, jitter_y, duration=0.15)

    def click(self, x: int, y: int, label: str = "") -> None:
        self._throttle()
        # 点击前轻微抖动，避免每次落在完全相同的像素
        tx, ty = self._global(x + random.randint(-2, 2), y + random.randint(-2, 2))
        if self.dry_run or self._gui is None:
            self.log.info("[dry-run] 点击 (%d,%d) %s", tx, ty, label)
            return
        self.log.info("点击 (%d,%d) %s", tx, ty, label)
        self._safe(self._gui.moveTo, tx, ty, duration=random.uniform(0.18, 0.35))
        self._safe(self._gui.click)

    def click_rect(self, rect: Rect, label: str = "") -> None:
        self.click(rect.cx, rect.cy, label)

    def drag(self, x0: int, y0: int, x1: int, y1: int, label: str = "") -> None:
        if self.dry_run:
            self.log.info("拖动 (%d,%d) -> (%d,%d) %s", x0, y0, x1, y1, label)
            return
        gui = self._gui
        if gui is None:
            return
        gx0, gy0 = self._global(x0, y0)
        gx1, gy1 = self._global(x1, y1)
        self._safe(gui.moveTo, gx0, gy0)
        self._safe(gui.mouseDown)
        self._safe(gui.moveTo, gx1, gy1, 0.35)
        self._safe(gui.mouseUp)
        self.log.info("拖动 (%d,%d) -> (%d,%d) %s", x0, y0, x1, y1, label)

    def scroll_at(self, x: int, y: int, clicks: int) -> None:
        self._throttle()
        gx, gy = self._global(x, y)
        if self.dry_run or self._gui is None:
            self.log.info("[dry-run] 在 (%d,%d) 滚动 %d", gx, gy, clicks)
            return
        # 连续滚动时鼠标通常已经停在目录上，省掉每次 0.12s 的移动
        current = self._safe(self._gui.position)
        if current is None or abs(int(current.x) - gx) > 8 or abs(int(current.y) - gy) > 8:
            self._safe(self._gui.moveTo, gx, gy, duration=0.12)
        self._safe(self._gui.scroll, int(clicks))
        time.sleep(0.03)

    def press(self, key: str) -> None:
        self._throttle()
        if self.dry_run or self._gui is None:
            self.log.info("[dry-run] 按键 %s", key)
            return
        self._safe(self._gui.press, key)

    def hotkey(self, *keys: str) -> None:
        """组合键（如 ctrl+w 关闭练习标签页）。"""
        self._throttle()
        if self.dry_run or self._gui is None:
            self.log.info("[dry-run] 组合键 %s", "+".join(keys))
            return
        self._safe(self._gui.hotkey, *keys)
