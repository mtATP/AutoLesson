from __future__ import annotations

import logging
import re
import time
from typing import Optional

import numpy as np

from . import vision
from .config import Config
from .layout import Layout
from .ocr import OcrEngine
from .types import PlaybackState, Rect

log = logging.getLogger("autolesson.player")

TIME_RANGE_RE = re.compile(
    r"(\d{1,2}\s*:\s*\d{2}(?:\s*:\s*\d{2})?)\s*/\s*(\d{1,2}\s*:\s*\d{2}(?:\s*:\s*\d{2})?)"
)


def parse_clock(text: str) -> Optional[float]:
    parts = [p for p in re.split(r"\s*:\s*", text.strip()) if p]
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return None


def parse_time_range(text: str) -> Optional[tuple[float, float]]:
    text = (text or "").replace("：", ":").replace("／", "/")
    m = TIME_RANGE_RE.search(text)
    if not m:
        return None
    current = parse_clock(m.group(1))
    total = parse_clock(m.group(2))
    if current is None or total is None or total <= 0:
        return None
    return current, total


class PlayerMonitor:
    def __init__(self, ocr: OcrEngine, cfg: Config, logger: Optional[logging.Logger] = None):
        self.ocr = ocr
        self.cfg = cfg
        self.log = logger or log

    # ------------------------------------------------------------------ 状态
    def _read_time_range(
        self,
        frame: np.ndarray,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        upscale: float,
    ) -> tuple[Optional[tuple[float, float]], Optional[Rect]]:
        h, w = frame.shape[:2]
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 - x0 < 40 or y1 - y0 < 15:
            return None, None
        strip = frame[y0:y1, x0:x1]
        try:
            lines = self.ocr.read(strip, offset=(x0, y0), upscale=upscale, min_score=0.30)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("时间文字 OCR 失败: %s", exc)
            lines = []
        parsed = None
        time_rect: Optional[Rect] = None
        for line in lines:
            value = parse_time_range(line.text)
            if value is not None:
                parsed = value
                time_rect = line.rect
                break
        if parsed is None and lines:
            joined = " ".join(line.text for line in lines)
            parsed = parse_time_range(joined)
            if parsed is not None:
                rects = [ln.rect for ln in lines if ":" in ln.text]
                if rects:
                    time_rect = rects[0]
                    for rect in rects[1:]:
                        time_rect = time_rect.union(rect)
        return parsed, time_rect

    def snapshot(self, frame: np.ndarray, layout: Layout) -> Optional[PlaybackState]:
        h, w = frame.shape[:2]
        # 默认排除屏幕最底部 5%（Windows 任务栏），避免任务栏亮色/图标被当成进度条
        margin = int(max(0.0, float(getattr(self.cfg.player, "bottom_margin_ratio", 0.05))) * h)
        bar = vision.find_progress_bar(frame, layout.panel_left, bottom_margin=margin)
        pw = layout.player_width
        upscale = 2.0 if pw < 1500 else 1.5

        parsed = None
        time_rect: Optional[Rect] = None
        if bar is not None:
            parsed, time_rect = self._read_time_range(
                frame,
                int(0.03 * pw),
                int(0.42 * pw),
                bar.y - int(0.012 * pw),
                bar.y + int(0.062 * pw),
                upscale,
            )

        # 蓝色幻灯片会把整屏判成蓝色，find_progress_bar 找不到真正的细进度条；
        # 此时用左下角固定位置的时间文字兜底，仍能拿到 current/total 与暂停状态。
        if parsed is None:
            parsed, time_rect = self._read_time_range(
                frame,
                0,
                int(0.42 * pw),
                int(0.84 * h),
                int(0.99 * h),
                upscale,
            )
        if parsed is None:
            return None

        if bar is not None:
            bar_y = bar.y
            playing, button = vision.detect_play_state(frame, bar.y, pw, time_rect)
        else:
            bar_y = time_rect.cy if time_rect is not None else int(0.90 * h)
            playing, button = vision.detect_play_state(frame, bar_y, pw, time_rect)

        current, total = parsed
        ratio = max(0.0, min(1.0, current / total)) if total else (bar.ratio if bar else None)
        raw = f"{_fmt(current)}/{_fmt(total)}"
        return PlaybackState(
            current=current,
            total=total,
            ratio=ratio,
            playing=playing,
            bar_y=bar_y,
            play_button=button,
            time_rect=time_rect,
            raw_time=raw,
        )

    # ------------------------------------------------------------------ 控件
    def show_controls(self, mouse, layout: Layout) -> None:
        """把鼠标移到播放器底部控制条区域，唤出进度条；不点击，避免暂停/隐藏控件。"""
        if mouse is None or not bool(self.cfg.player.controls_wiggle):
            return
        x, y = layout.player_center
        try:
            mouse.move_to(x, y, duration=0.15)
            time.sleep(0.12)
            mouse.wiggle(x, y)
        except Exception:  # noqa: BLE001
            try:
                mouse.wiggle(x, y)
            except Exception:  # noqa: BLE001
                pass

    def refresh(
        self,
        screen,
        layout: Layout,
        mouse=None,
        attempts: int = 3,
    ) -> Optional[PlaybackState]:
        """唤出控件并读取播放状态。"""
        for index in range(max(1, attempts)):
            frame = screen.grab()
            state = self.snapshot(frame, layout)
            if state is not None:
                return state
            self.show_controls(mouse, layout)
            time.sleep(0.55)
            if index == 0:
                self.log.debug("未找到进度条，移动鼠标唤出控件")
        return None

    def click_play(self, mouse, state: PlaybackState, label: str = "播放") -> None:
        if state.play_button is not None:
            mouse.click_rect(state.play_button, label)
        else:
            self.log.warning("未定位到播放按钮，跳过点击")

    def wait_for_advance(
        self,
        screen,
        layout: Layout,
        baseline: Optional[float],
        timeout: float = 10.0,
        mouse=None,
    ) -> Optional[PlaybackState]:
        deadline = time.monotonic() + max(1.0, timeout)
        last: Optional[PlaybackState] = None
        while time.monotonic() < deadline:
            state = self.refresh(screen, layout, mouse, attempts=1)
            if state is not None:
                last = state
                if baseline is None:
                    if state.ratio is not None and state.ratio > 0.0005:
                        return state
                elif state.current is not None and state.current > baseline + 0.4:
                    return state
            time.sleep(0.8)
        return last

    def video_center(self, layout: Layout) -> tuple[int, int]:
        return int(layout.panel_left * 0.5), int(layout.screen_h * 0.5)


def _fmt(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
