from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .types import (
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    ProgressBar,
    Rect,
)

log = logging.getLogger("autolesson.vision")


# --------------------------------------------------------------------------- 颜色
def blue_mask(img: np.ndarray) -> np.ndarray:
    """智慧树主题蓝（BGR）。"""
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    # 弹窗打开时整个页面会被压暗，阈值需要略低才能识别背景里的蓝色状态图标
    return (b > 105) & (b - r > 30) & (b - g > 10)


def light_mask(img: np.ndarray, threshold: int = 150) -> np.ndarray:
    if img.ndim == 2:
        return img >= threshold
    return np.minimum(np.minimum(img[:, :, 0], img[:, :, 1]), img[:, :, 2]) >= threshold


def white_mask(img: np.ndarray, threshold: int = 245) -> np.ndarray:
    return light_mask(img, threshold)


def mean_brightness(img: np.ndarray) -> float:
    if img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(gray.mean())


def is_black(img: np.ndarray, threshold: float = 8.0) -> bool:
    return mean_brightness(img) < threshold


# --------------------------------------------------------------------------- 进度条
def _split_groups(indices: np.ndarray) -> list[list[int]]:
    if len(indices) == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        value = int(value)
        if value - groups[-1][-1] <= 2:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def find_progress_bar(
    frame: np.ndarray,
    panel_left: int,
    bottom_margin: int = 0,
) -> Optional[ProgressBar]:
    """在播放器底部寻找进度条。

    判定条件：一行中 80% 以上为浅色轨道或蓝色已播部分，且至少有两个主题蓝像素
    （播放头/已播部分），并且位于画面下半部。控件隐藏时返回 None。

    bottom_margin 用于排除屏幕最底部的 Windows 任务栏：任务栏的亮色/蓝色图标
    会被误认成进度条，导致真正的播放器控制条被忽略。
    """
    h, w = frame.shape[:2]
    x_right = int(min(max(0, panel_left), w))
    if x_right < 80 or h < 80:
        return None
    y_start = int(h * 0.45)
    y_end = max(y_start, h - max(0, int(bottom_margin)))
    region = frame[:, :x_right]
    light = light_mask(region, 150)
    blue = blue_mask(region)
    light_counts = light.sum(axis=1)
    blue_counts = blue.sum(axis=1)
    rows = np.arange(h)
    # 进度条 = 浅色未播轨道 + 蓝色已播部分。播放到 70% 时浅色只剩 30%，
    # 不能再要求“80% 浅色”，否则高进度视频会被判定为没有进度条。
    track_counts = light_counts + blue_counts
    cond = (
        (track_counts >= 0.80 * x_right)
        & (blue_counts >= 2)
        & (rows >= y_start)
        & (rows < y_end)
    )
    candidates = np.nonzero(cond)[0]
    max_group = max(12, int(0.12 * h))
    groups = [g for g in _split_groups(candidates) if len(g) <= max_group]
    if not groups:
        return None

    # 同一组里播放头会向上下各延伸十几像素；用蓝色像素最多的一行定位轨道。
    # 进度条永远在播放器底部；优先取最靠下的一组，避免白色幻灯片上的蓝色元素干扰
    group = max(groups, key=lambda g: (g[-1], int(blue_counts[g].max())))
    peak = int(blue_counts[group].max())
    bar_rows = [y for y in group if blue_counts[y] >= max(2, 0.5 * peak)]
    if not bar_rows:
        bar_rows = group
    y = int(round(float(np.mean(bar_rows))))
    band = blue[group[0] : group[-1] + 1]
    if not band.any():
        return None
    xs = np.nonzero(band.any(axis=0))[0]
    if len(xs) == 0:
        return None

    # 真正的进度条一定有一个比轨道高得多的圆形播放头；纯蓝色横线（PPT 装饰线）不算
    window = max(8, int(0.025 * x_right))
    win = blue[max(0, y - window) : min(h, y + window)]
    max_extent = 0
    for col in range(win.shape[1]):
        ys = np.nonzero(win[:, col])[0]
        if len(ys):
            max_extent = max(max_extent, int(ys.max() - ys.min() + 1))
    if max_extent < max(5, int(0.004 * x_right)):
        return None

    playhead_max = int(xs.max())
    # 播放头是直径约 1.5% 宽度的圆点，把圆点半径从最右端扣掉。
    radius = max(2.0, 0.008 * x_right)
    playhead_center = max(0.0, playhead_max - radius)
    ratio = float(np.clip(playhead_center / x_right, 0.0, 1.0))
    return ProgressBar(
        y=y,
        x_right=x_right,
        playhead_x=int(playhead_center),
        ratio=ratio,
        rows=tuple(group),
    )


# --------------------------------------------------------------------------- 播放/暂停
def detect_play_state(
    frame: np.ndarray,
    bar_y: int,
    player_width: int,
    time_rect: Optional[Rect] = None,
) -> tuple[Optional[bool], Optional[Rect]]:
    """通过按钮形状判断播放状态。

    返回 (playing, 按钮矩形)。playing=True 表示正在播放（暂停图标为两条竖线），
    False 表示已暂停（三角形播放图标），None 表示无法判断。
    """
    h, w = frame.shape[:2]
    if time_rect is not None:
        x0 = 0
        x1 = max(0, time_rect.x0 - max(6, int(0.012 * player_width)))
        y0 = time_rect.cy - int(0.022 * player_width)
        y1 = time_rect.cy + int(0.022 * player_width)
    else:
        x0 = 0
        x1 = int(0.10 * player_width)
        y0 = bar_y + int(0.006 * player_width)
        y1 = bar_y + int(0.050 * player_width)
    x0 = max(0, x0)
    y0 = max(0, y0, bar_y + int(0.004 * player_width))
    x1 = min(w, x1)
    y1 = min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None

    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 局部对比度：只保留“比周围更亮”的图标，避免浅色控制条背景干扰。
    background = cv2.GaussianBlur(gray, (0, 0), 4)
    contrast = gray.astype(np.int16) - background.astype(np.int16)
    mask = (contrast > 30).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    min_h = max(5, int(0.006 * player_width))
    max_h = max(12, int(0.045 * player_width))
    comps: list[tuple[Rect, int]] = []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area < 9 or ch < min_h or ch > max_h:
            continue
        if cw > 0.035 * player_width:
            continue
        if cw / max(1, ch) < 0.10 or cw / max(1, ch) > 1.7:
            continue
        # 播放/暂停键在控制条最左侧。下一节按钮（三角+竖线）在它右边，
        # 不能把它当播放键点，否则会直接切到下一视频。
        if x0 + x + cw > max(24, int(0.03 * player_width)):
            continue
        comps.append((Rect(x0 + x, y0 + y, x0 + x + cw, y0 + y + ch), area))
    if not comps:
        return None, None

    comps.sort(key=lambda item: item[0].x0)
    clusters: list[list[tuple[Rect, int]]] = [[comps[0]]]
    gap_limit = max(6, int(0.012 * player_width))
    for item in comps[1:]:
        if item[0].x0 - clusters[-1][-1][0].x1 <= gap_limit:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    cluster = clusters[0]
    if len(cluster) > 3:
        return None, None
    union = cluster[0][0]
    for rect, _area in cluster[1:]:
        union = union.union(rect)
    if len(cluster) == 2:
        # 两条等高的竖线 = 暂停按钮 = 正在播放
        (a, _), (b, _) = cluster
        if abs(a.height - b.height) <= max(3, 0.25 * max(a.height, b.height)):
            return True, union.expand(5).clamp(w, h)
    if len(cluster) == 1:
        rect, area = cluster[0]
        fill = area / max(1, rect.area)
        aspect = rect.width / max(1, rect.height)
        # 两条竖线被 3x3 闭运算连成一个 22x18、填充率约 0.22 的连通域
        if aspect > 1.05 and fill < 0.42:
            return True, rect.expand(5).clamp(w, h)
        # 单个三角形播放按钮：宽高比接近 0.6~0.8，填充率约 0.4~0.6
        if aspect <= 1.15 and 0.30 <= fill <= 0.80:
            return False, rect.expand(5).clamp(w, h)
        return None, rect.expand(5).clamp(w, h)
    return None, union.expand(5).clamp(w, h)


# --------------------------------------------------------------------------- 状态图标
def find_status_icon(
    frame: np.ndarray,
    zone: Rect,
    row_top: int,
    row_bottom: int,
    font_h: int,
    anchor_y: Optional[int] = None,
) -> tuple[Optional[str], Optional[Rect]]:
    """在目录行右侧寻找蓝色状态图标。

    实心蓝圆（内部蓝色比例高）= 已完成；空心蓝环 = 学习中。
    """
    h, w = frame.shape[:2]
    x0 = max(0, zone.x0)
    x1 = min(w, zone.x1)
    y0 = max(0, row_top)
    y1 = min(h, row_bottom)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None
    roi = frame[y0:y1, x0:x1]
    mask = blue_mask(roi).astype(np.uint8)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    best: Optional[tuple[int, Rect, np.ndarray]] = None
    min_size = max(8, int(0.8 * font_h))
    max_size = max(18, int(2.2 * font_h))
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area < 25:
            continue
        if cw < min_size or ch < min_size or cw > max_size or ch > max_size:
            continue
        if cw / max(1, ch) < 0.55 or cw / max(1, ch) > 1.8:
            continue
        rect = Rect(x0 + x, y0 + y, x0 + x + cw, y0 + y + ch)
        # 右侧悬浮的“AI助教”头像也是蓝色圆形，用垂直位置排除
        if anchor_y is not None and abs(rect.cy - anchor_y) > 0.9 * font_h:
            continue
        if best is None or area > best[0]:
            best = (area, rect, mask[y : y + ch, x : x + cw])
    if best is None:
        return None, None

    _area, rect, sub = best
    sh, sw = sub.shape[:2]
    yy, xx = np.mgrid[0:sh, 0:sw]
    cx, cy = (sw - 1) / 2.0, (sh - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = radius <= 0.35 * min(sw, sh) / 2.0
    inner_ratio = float(sub[inner].mean()) if inner.any() else 0.0
    if inner_ratio >= 0.35:
        return STATUS_COMPLETED, rect
    return STATUS_IN_PROGRESS, rect


# --------------------------------------------------------------------------- 弹窗卡片
def find_cards(
    frame: np.ndarray,
    exclude_x: int,
    threshold: int = 225,
    close_kernel: int = 15,
    max_cards: int = 3,
) -> list[Rect]:
    """寻找播放器区域里的大块浅色卡片（弹题、公告、学前必读等）。"""
    h, w = frame.shape[:2]
    mask = white_mask(frame, threshold).astype(np.uint8)
    if exclude_x > 0:
        mask[:, min(exclude_x, w) :] = 0
    k = max(3, int(close_kernel) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    cards: list[Rect] = []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if cw < 0.15 * w or ch < 0.12 * h:
            continue
        if area < 0.02 * w * h:
            continue
        rect = Rect(x, y, x + cw, y + ch)
        # 全屏纯白画面（白色 PPT 铺满播放器）不是弹窗
        if x <= 2 and y <= 2 and cw >= 0.95 * w and ch >= 0.80 * h:
            continue
        # 浏览器/页面顶部工具条：贴着顶边、很宽、很扁
        if y <= int(0.16 * h) and ch < int(0.26 * h) and cw > 0.80 * max(1, exclude_x if 0 < exclude_x < w else w):
            continue
        cards.append(rect)
    cards.sort(key=lambda r: r.area, reverse=True)
    return cards[: max(1, int(max_cards))]


# --------------------------------------------------------------------------- 其它
def image_signature(img: np.ndarray) -> bytes:
    """任意图像的快速 dHash 指纹。"""
    if img is None or img.size == 0:
        return b""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (32, 96), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return np.packbits(diff).tobytes()


def panel_signature(frame: np.ndarray, panel: Rect) -> bytes:
    """目录区域快速指纹，用于判断滚动是否生效。"""
    return image_signature(panel.crop(frame))


def panel_scroll_signature(frame: np.ndarray, panel: Rect, right_margin: int = 62) -> bytes:
    """滚动指纹：排除右侧状态图标/滚动条，避免加载动画被误判成滚动。"""
    crop = panel.crop(frame)
    margin = max(0, int(right_margin))
    if crop.shape[1] > margin + 20:
        crop = crop[:, : crop.shape[1] - margin]
    return image_signature(crop)


def signature_distance(a: bytes, b: bytes) -> int:
    if not a or not b or len(a) != len(b):
        return 10**6
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def find_directory_label(lines: list, frame_w: int) -> Optional[Rect]:
    """在 OCR 结果中寻找“目录”二字，用于自动定位右侧面板。"""
    for line in lines:
        text = line.text.replace(" ", "")
        if "目录" in text and line.cx > 0.55 * frame_w:
            return line.rect
    return None
