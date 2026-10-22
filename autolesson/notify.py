from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests

from .config import Config

log = logging.getLogger("autolesson.notify")


class Notifier:
    """控制台 + 桌面 + 声音 + Webhook 通知。任何通知失败都不会中断主流程。"""

    def __init__(self, cfg: Config, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.log = logger or log
        self._last: dict[str, float] = {}

    def send(
        self,
        title: str,
        message: str,
        level: str = "info",
        shot: Optional[str | Path] = None,
        force: bool = False,
    ) -> None:
        text = message.strip()
        if shot:
            text += f"\n截图: {shot}"
        dedup_key = f"{level}|{title}|{text}"
        now = time.time()
        if not force:
            last = self._last.get(dedup_key, 0.0)
            if now - last < float(self.cfg.notify.dedup_seconds):
                return
        self._last[dedup_key] = now

        log_fn = self.log.error if level in ("error", "fatal") else self.log.info
        log_fn("\n%s\n[%s] %s\n%s\n%s", "=" * 72, level.upper(), title, text, "=" * 72)

        if bool(self.cfg.notify.desktop):
            self._desktop(title, text)
        if bool(self.cfg.notify.sound):
            self._sound(level)
        self._webhook(title, text, level, shot)

    # ------------------------------------------------------------------ 桌面
    def _desktop(self, title: str, message: str) -> None:
        try:
            from plyer import notification  # type: ignore

            notification.notify(title=title, message=message[:240], app_name="AutoLesson", timeout=8)
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            system = platform.system()
            if system == "Darwin":
                safe_title = title.replace('"', "'")
                safe_msg = message[:240].replace('"', "'")
                subprocess.Popen(
                    ["osascript", "-e", f'display notification "{safe_msg}" with title "{safe_title}"'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif system == "Linux" and shutil.which("notify-send"):
                subprocess.Popen(
                    ["notify-send", title, message[:240]],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:  # noqa: BLE001
            self.log.debug("桌面通知失败", exc_info=True)

    # ------------------------------------------------------------------ 声音
    def _sound(self, level: str) -> None:
        def beep() -> None:
            try:
                if sys.platform == "win32":
                    import winsound  # type: ignore

                    if level in ("error", "fatal"):
                        winsound.Beep(880, 350)
                        winsound.Beep(660, 500)
                    else:
                        winsound.MessageBeep(0x40)
                elif sys.platform == "darwin":
                    subprocess.Popen(
                        ["afplay", "/System/Library/Sounds/Glass.aiff"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    if shutil.which("paplay"):
                        subprocess.Popen(
                            ["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        print("\a", end="", flush=True)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=beep, daemon=True).start()

    # ------------------------------------------------------------------ Webhook
    def _webhook(self, title: str, message: str, level: str, shot: Optional[str | Path]) -> None:
        url = str(self.cfg.notify.webhook_url or "").strip()
        kind = str(self.cfg.notify.webhook_type or "generic").lower()
        if kind == "telegram":
            token = str(self.cfg.notify.telegram_bot_token or "").strip()
            chat_id = str(self.cfg.notify.telegram_chat_id or "").strip()
            if not token or not chat_id:
                self.log.debug("Telegram 通知未配置 token/chat_id")
                return
            try:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": f"{title}\n\n{message[:3500]}"},
                    timeout=10,
                )
            except Exception as exc:  # noqa: BLE001
                self.log.warning("Telegram 通知失败: %s", exc)
            return
        if not url:
            return
        try:
            if kind == "serverchan":
                target = url
                if not target.startswith("http"):
                    target = f"https://sctapi.ftqq.com/{target}.send"
                requests.post(target, data={"title": title, "desp": message[:3500]}, timeout=10)
            else:
                requests.post(
                    url,
                    json={
                        "title": title,
                        "message": message[:3500],
                        "level": level,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "host": platform.node(),
                        "screenshot": str(shot) if shot else None,
                    },
                    timeout=10,
                )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Webhook 通知失败: %s", exc)
