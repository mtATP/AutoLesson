from __future__ import annotations

import ctypes
import logging
import sys
import time

log = logging.getLogger("autolesson.platform")


def enable_dpi_awareness() -> None:
    """让截图坐标与鼠标坐标在 Windows 高 DPI 下保持一致。"""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            log.debug("无法开启 DPI 感知", exc_info=True)


def get_active_window_title() -> str:
    """返回当前前台窗口标题（Chrome 中即当前标签页标题）。"""
    if sys.platform != "win32":
        return ""
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = user32.GetForegroundWindow()
        length = int(user32.GetWindowTextLengthW(hwnd))
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:  # noqa: BLE001
        return ""


def focus_window(keywords: list[str], logger: logging.Logger | None = None) -> bool:
    """按标题关键字激活浏览器窗口。依赖可选的 pygetwindow。"""
    logger = logger or log
    try:
        import pygetwindow as gw  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.debug("pygetwindow 不可用，跳过窗口聚焦: %s", exc)
        return False
    try:
        windows = gw.getAllWindows()
    except Exception as exc:  # noqa: BLE001
        logger.debug("枚举窗口失败: %s", exc)
        return False
    for keyword in keywords or []:
        for win in windows:
            try:
                title = win.title or ""
                if not title or not win.visible:
                    continue
                if keyword.lower() not in title.lower():
                    continue
                if win.isMinimized:
                    win.restore()
                    time.sleep(0.3)
                win.activate()
                time.sleep(0.6)
                logger.info("已聚焦窗口: %s", title)
                return True
            except Exception:  # noqa: BLE001
                continue
    logger.warning("未找到包含 %s 的浏览器窗口，请手动保持课程页面在前台", keywords)
    return False
