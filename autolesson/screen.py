from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .platform_utils import enable_dpi_awareness
from .types import Rect


class ScreenUnavailable(RuntimeError):
    pass


class Screen:
    """mss 截屏封装，统一输出 BGR ndarray。"""

    def __init__(self, monitor: int = 1, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger("autolesson.screen")
        enable_dpi_awareness()
        try:
            import mss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ScreenUnavailable("缺少 mss，请执行: pip install mss") from exc
        try:
            self._sct = mss.mss()
        except Exception as exc:  # noqa: BLE001
            raise ScreenUnavailable(
                "无法截屏。请确认已登录桌面会话且 DISPLAY 正常（远程/锁屏/无头环境会失败）"
            ) from exc
        index = int(monitor)
        if index < 1 or index >= len(self._sct.monitors):
            index = 1
        self.monitor = index
        self._mon = self._sct.monitors[index]
        self.width = int(self._mon["width"])
        self.height = int(self._mon["height"])
        self.origin = (int(self._mon["left"]), int(self._mon["top"]))
        self.log.info(
            "截屏区域: monitor=%d %dx%d @ (%d,%d)",
            index,
            self.width,
            self.height,
            self.origin[0],
            self.origin[1],
        )

    def grab(self) -> np.ndarray:
        shot = self._sct.grab(self._mon)
        arr = np.asarray(shot)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ScreenUnavailable("截屏返回了非预期的图像格式")
        return np.ascontiguousarray(arr[:, :, :3])

    def grab_region(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        frame = self.grab()
        return frame[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]

    def grab_rect(self, rect: Rect) -> np.ndarray:
        """只截取指定区域，用于目录滚动的快速指纹，避免每次复制整屏。"""
        x0 = max(0, min(int(rect.x0), self.width - 1))
        y0 = max(0, min(int(rect.y0), self.height - 1))
        x1 = max(x0 + 1, min(int(rect.x1), self.width))
        y1 = max(y0 + 1, min(int(rect.y1), self.height))
        region = {
            "left": self.origin[0] + x0,
            "top": self.origin[1] + y0,
            "width": x1 - x0,
            "height": y1 - y0,
        }
        shot = self._sct.grab(region)
        arr = np.asarray(shot)
        return np.ascontiguousarray(arr[:, :, :3])

    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def save(path: Path | str, frame: np.ndarray) -> bool:
        """保存截图，兼容中文路径。"""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                return False
            buf.tofile(str(path))
            return True
        except Exception:  # noqa: BLE001
            logging.getLogger("autolesson.screen").warning("保存截图失败: %s", path, exc_info=True)
            return False
