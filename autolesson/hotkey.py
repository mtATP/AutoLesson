"""全局停止热键。

Windows 下使用系统原生 RegisterHotKey，浏览器/播放器在前台时也能按。
不引入额外第三方依赖；非 Windows 环境自动降级为提示使用 Ctrl+C。
"""

from __future__ import annotations

import ctypes
import sys
import threading
from typing import Any, Optional

MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
LLKHF_INJECTED = 0x10

_VK = {f"F{i}": 0x6F + i for i in range(1, 13)}
_VK.update({"ESC": 0x1B, "ESCAPE": 0x1B, "PAUSE": 0x13, "HOME": 0x24, "END": 0x23})


def parse_hotkey(key: str) -> Optional[int]:
    """把 F9/ESC 等配置值转换成 Windows 虚拟键码；不支持返回 None。"""
    if key is None:
        return None
    return _VK.get(str(key).strip().upper())


class HotkeyListener:
    """注册一个系统级停止热键，按下后设置 stop_event。"""

    HOTKEY_ID = 0x4153

    def __init__(
        self,
        key: str,
        stop_event: threading.Event,
        log: Any,
        extra_keys: Optional[list[str]] = None,
    ):
        self.key = str(key or "F9").strip().upper()
        self.extra_keys = [
            str(k).strip().upper()
            for k in (extra_keys or [])
            if str(k).strip() and str(k).strip().upper() != self.key
        ]
        self.keys = [self.key] + self.extra_keys
        self.stop_event = stop_event
        self.log = log
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._registered = False
        self._ready = threading.Event()
        self._hook_thread: Optional[threading.Thread] = None
        self._hook_thread_id: Optional[int] = None
        self._hook = None
        self._hook_proc = None

    def start(self) -> None:
        if sys.platform != "win32":
            self.log.info("全局停止热键仅支持 Windows；当前请用 Ctrl+C 停止")
            return
        if parse_hotkey(self.key) is None:
            self.log.warning(
                "无法识别停止热键 %r，已禁用（支持 F1~F12/ESC/PAUSE）", self.key
            )
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="autolesson-hotkey"
        )
        self._thread.start()
        # 额外按键（默认 ESC）用低级键盘钩子监听，但不拦截按键：
        # 这样脚本自己按 ESC 关闭弹题/图片查看器时不会反过来停止程序。
        valid_extra = [k for k in self.extra_keys if parse_hotkey(k) is not None]
        if valid_extra:
            self._hook_thread = threading.Thread(
                target=self._run_extra_hook,
                args=(valid_extra,),
                daemon=True,
                name="autolesson-hotkey-extra",
            )
            self._hook_thread.start()

    def stop(self) -> None:
        if sys.platform == "win32":
            self._ready.wait(timeout=0.5)
            if self._thread_id:
                try:
                    ctypes.windll.user32.PostThreadMessageW(
                        self._thread_id, WM_QUIT, 0, 0
                    )
                except Exception:  # noqa: BLE001
                    pass
            if self._hook_thread_id:
                try:
                    ctypes.windll.user32.PostThreadMessageW(
                        self._hook_thread_id, WM_QUIT, 0, 0
                    )
                except Exception:  # noqa: BLE001
                    pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        hook_thread = self._hook_thread
        if hook_thread is not None and hook_thread.is_alive():
            hook_thread.join(timeout=1.0)

    def _on_hotkey(self, key: Optional[str] = None) -> None:
        if not self.stop_event.is_set():
            self.log.warning("收到停止热键 %s，正在安全停止...", key or self.key)
            self.stop_event.set()

    def _run_extra_hook(self, keys: list[str]) -> None:
        """低级键盘钩子：物理按 ESC 停止，但不吞按键，也不响应脚本注入的按键。"""
        from ctypes import wintypes

        class KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ("vkCode", wintypes.DWORD),
                ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_void_p),
            ]

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.SetWindowsHookExW.restype = ctypes.c_void_p
        user32.CallNextHookEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.CallNextHookEx.restype = ctypes.c_ssize_t
        user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
        vk_to_key = {
            int(parse_hotkey(k)): k for k in keys if parse_hotkey(k) is not None
        }
        if not vk_to_key:
            return
        self._hook_thread_id = int(kernel32.GetCurrentThreadId())
        HOOKPROC = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
        )

        def proc(n_code, w_param, l_param):
            try:
                if n_code == 0 and int(w_param) in (WM_KEYDOWN, WM_SYSKEYDOWN):
                    info = ctypes.cast(
                        l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)
                    ).contents
                    if not (int(info.flags) & LLKHF_INJECTED):
                        key = vk_to_key.get(int(info.vkCode))
                        if key:
                            self._on_hotkey(key)
            except Exception:  # noqa: BLE001
                pass
            return user32.CallNextHookEx(0, n_code, w_param, l_param)

        self._hook_proc = HOOKPROC(proc)
        self._hook = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL,
            self._hook_proc,
            None,
            0,
        )
        if not self._hook:
            self.log.warning("ESC 全局监听注册失败（可能被安全软件拦截）")
            return
        self.log.info(
            "全局停止热键已启用：%s（ESC 监听不会拦截浏览器按键）",
            "、".join(self.keys),
        )
        msg = wintypes.MSG()
        try:
            while not self.stop_event.is_set():
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret <= 0:
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception as exc:  # noqa: BLE001
            self.log.warning("ESC 全局监听异常: %s", exc)
        finally:
            if self._hook:
                try:
                    user32.UnhookWindowsHookEx(self._hook)
                except Exception:  # noqa: BLE001
                    pass
                self._hook = None

    def _run(self) -> None:
        from ctypes import wintypes

        vk = parse_hotkey(self.key)
        if vk is None:
            return
        user32 = ctypes.windll.user32
        self._thread_id = int(ctypes.windll.kernel32.GetCurrentThreadId())
        if not user32.RegisterHotKey(None, self.HOTKEY_ID, MOD_NOREPEAT, vk):
            self.log.warning(
                "全局停止热键 %s 注册失败（可能被其他程序占用）；"
                "请用 Ctrl+C 或修改 runtime.stop_hotkey",
                self.key,
            )
            return
        self._registered = True
        self._ready.set()
        self.log.info(
            "全局停止热键已启用：按 %s 可随时安全停止（浏览器前台也有效）", self.key
        )
        msg = wintypes.MSG()
        try:
            while not self.stop_event.is_set():
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret <= 0:
                    break
                if msg.message == WM_HOTKEY:
                    self._on_hotkey(self.key)
                    break
        except Exception as exc:  # noqa: BLE001
            self.log.warning("全局停止热键异常: %s", exc)
        finally:
            if self._registered:
                try:
                    user32.UnregisterHotKey(None, self.HOTKEY_ID)
                except Exception:  # noqa: BLE001
                    pass
                self._registered = False
