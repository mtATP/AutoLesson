"""泛雅视频挂机控制器（第一版：只处理视频任务点）。"""

from __future__ import annotations

import hashlib
import itertools
import logging
import re
import time
from typing import Optional

import cv2
import numpy as np

from autolesson.config import Config
from autolesson.types import FatalError, QuizQuestion, Rect, UserAbort, normalize_title

from .catalog import FanyaCatalogScanner
from .layout import FanyaLayout
from .navigator import FanyaCatalogNavigator
from .page import FanyaPageInfo, FanyaPageReader
from .practice import FanyaPracticeSolver
from .types import STATUS_COMPLETED, STATUS_PENDING, FanyaItem
from .video_quiz import VideoQuiz, VideoQuizReader

log = logging.getLogger("fanya.controller")

STOP_WORDS = ("登录", "验证码", "人脸识别", "拖动滑块", "身份验证")
PROMPT_WORDS = ("继续学习", "继续播放", "继续上次播放")
COMPLETION_GRACE = 15.0


def _norm(text: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    return text.replace("（", "(").replace("）", ")").replace("、", "")


def _title_matches(a: str, b: str, prefix: int = 8) -> bool:
    left, right = _norm(a), _norm(b)
    if not left or not right:
        return False
    return left.startswith(right[:prefix]) or right.startswith(left[:prefix])


class FanyaWatcher:
    def __init__(
        self,
        cfg: Config,
        ocr,
        screen,
        mouse,
        logger=None,
        dry_run: bool = False,
        state=None,
        llm=None,
        stop_event=None,
        mode: str = "video",
    ):
        self.cfg = cfg
        self.ocr = ocr
        self.screen = screen
        self.mouse = mouse
        self.log = logger or log
        self.dry_run = bool(dry_run)
        self.state = state
        self.llm = llm
        self.stop_event = stop_event
        self.mode = mode if mode in ("video", "practice", "auto") else "video"
        self.quiz_reader = VideoQuizReader(ocr)
        self._video_quiz_attempts: dict[str, dict] = {}
        self.practice_solver = FanyaPracticeSolver(
            ocr,
            cfg,
            mouse,
            screen,
            llm,
            state,
            logger,
            stop_event=stop_event,
        )
        self.layout: Optional[FanyaLayout] = None
        self.scanner = FanyaCatalogScanner(ocr)
        self.reader = FanyaPageReader(ocr)
        self.navigator: Optional[FanyaCatalogNavigator] = None

    # ------------------------------------------------------------------ 初始化
    def calibrate(self, frame: Optional[np.ndarray] = None) -> FanyaLayout:
        frame = self.screen.grab() if frame is None else frame
        lines = self.ocr.read(frame)
        self.layout = FanyaLayout.detect(frame, lines)
        self.navigator = FanyaCatalogNavigator(
            self.scanner, self.screen, self.mouse, self.layout, self.cfg
        )
        return self.layout

    @staticmethod
    def _video_key(item: FanyaItem) -> str:
        return f"{item.index}|{normalize_title(item.title)}"

    def _logged_video(self, item: FanyaItem):
        if self.state is None:
            return None
        videos = self.state.data.get("videos")
        if not isinstance(videos, dict):
            return None
        return videos.get(self._video_key(item))

    def _mark_video(self, item: FanyaItem, source: str) -> None:
        """记录“视频任务点已完成”。泛雅目录在章节测验未完成时不打勾，靠这条记录。"""
        if self.state is None:
            return
        key = self._video_key(item)
        videos = self.state.data.setdefault("videos", {})
        if not isinstance(videos, dict):
            videos = {}
            self.state.data["videos"] = videos
        record = {
            "index": item.index,
            "title": item.title,
            "completed_at": time.time(),
            "completed_text": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
        }
        videos[key] = record
        self.state.save()
        self.log.info(
            "视频完成记录: %s %s（来源: %s；目录未打勾时以此记录/日志为准）",
            item.index,
            item.title,
            source,
        )

    def _page_to_top(self) -> None:
        """把课程页正文滚回顶部；标题/播放器位置按顶部布局标定。

        只滚正文左侧，不碰右侧目录的独立滚动条；鼠标滚轮放在标题/标签
        区域，避免落在视频画面上触发音量/进度。
        """
        x = int(self.layout.screen_w * 0.42) if self.layout is not None else 1000
        y = int(self.layout.screen_h * 0.14) if self.layout is not None else 200
        try:
            for _ in range(3):
                self.mouse.scroll_at(x, y, +10)
                time.sleep(0.2)
        except UserAbort:
            raise
        except Exception as exc:  # noqa: BLE001
            self.log.debug("页面滚到顶部失败: %s", exc)

    def plan(self, frame: Optional[np.ndarray] = None) -> list[FanyaItem]:
        if frame is None:
            self._page_to_top()
            frame = self.screen.grab()
        if self.layout is None:
            self.calibrate(frame)
        assert self.navigator is not None
        self.log.info("开始扫描泛雅目录...")
        items = self.navigator.scan_all()
        if self.mode == "practice":
            pending = [it for it in items if self._practice_pending(it)]
        elif self.mode == "video":
            pending = [it for it in items if self._video_pending(it)]
        else:
            pending = [
                it for it in items
                if self._video_pending(it) or self._practice_pending(it)
            ]
        self.log.info(
            "目录共 %d 个任务点，当前模式 %s，待处理 %d 个",
            len(items),
            {"video": "视频", "practice": "练习", "auto": "列表"}.get(self.mode, self.mode),
            len(pending),
        )
        return pending

    def _logged_video(self, item: FanyaItem):
        if self.state is None:
            return None
        videos = self.state.data.get("videos")
        if not isinstance(videos, dict):
            return None
        return videos.get(self._video_key(item))

    def _saved_practice(self, item: FanyaItem):
        if self.state is None:
            return None
        practices = self.state.data.get("practices")
        if not isinstance(practices, dict):
            return None
        record = practices.get(self._video_key(item))
        return record if isinstance(record, dict) and record.get("coverage_version") == 2 else None

    def _video_pending(self, item: FanyaItem) -> bool:
        if item.completed:
            return False
        # “1”表示视频/练习中完成其一；按用户约定，视频优先时视为视频已完成
        if item.pending_count == 1:
            return False
        if self._logged_video(item):
            return False
        return True

    def _practice_pending(self, item: FanyaItem) -> bool:
        if item.completed:
            return False
        # 橙色 pending 即使数字 OCR 失败也要处理；绿勾在合并时优先，
        # 因此这里不再要求必须读到 1/2。
        if item.status != STATUS_PENDING and item.pending_count not in (1, 2):
            return False
        if self._saved_practice(item):
            return False
        return True

    def _mark_practice(self, item: FanyaItem, source: str) -> None:
        if self.state is None:
            return
        key = self._video_key(item)
        practices = self.state.data.setdefault("practices", {})
        if not isinstance(practices, dict):
            practices = {}
            self.state.data["practices"] = practices
        practices[key] = {
            "index": item.index,
            "title": item.title,
            "coverage_version": 2,
            "saved_at": time.time(),
            "saved_text": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
        }
        self.state.save()
        self.log.info("练习保存记录: %s %s（来源: %s）", item.index, item.title, source)

    # ------------------------------------------------------------------ 主循环
    def run(self, max_items: int = 0) -> int:
        pending = self.plan()
        if self.dry_run:
            self.log.info("dry-run：只扫描，不点击")
            for item in pending[: max_items or None]:
                self.log.info("待处理: %s %s", item.index, item.title)
            return 0
        done = 0
        done_practice = 0
        for item in pending:
            self._check_stop()
            if max_items and done + done_practice >= max_items:
                break
            video_pending = self._video_pending(item)
            practice_pending = self._practice_pending(item)
            if self.mode == "practice" or (self.mode == "auto" and not video_pending):
                if practice_pending:
                    if self._practice_item(item):
                        done_practice += 1
                continue
            logged = self._logged_video(item)
            if logged:
                self.log.info(
                    "日志显示视频已完成（目录可能未打勾），打开确认: %s %s",
                    item.index,
                    item.title,
                )
            self.log.info("下一个任务点: %s %s", item.index, item.title)
            found = self.navigator.seek_forward(item.index) or self.navigator.seek_from_top(item.index)
            if found is None:
                self.log.warning("目录里找不到任务点 %s %s，跳过", item.index, item.title)
                continue
            if not self._open_item(found):
                self.log.warning("首次点击未生效，重新定位后再试: %s", item.index)
                again = self.navigator.seek_from_top(item.index)
                if again is None or not self._open_item(again):
                    raise FatalError("泛雅目录项无法点击", f"{item.index} {item.title}")
                found = again
            page = self._wait_page(found, timeout=10)
            if page is None:
                # 不继续点下一条：上一次实测中标题识别失败后连续点了多个
                # 目录项，最后播放了错误的视频。重定位同一条再试一次。
                self.log.warning("首次打开未确认页面，重新定位同一条再试一次: %s", found.title)
                again = self.navigator.seek_from_top(item.index)
                if again is not None:
                    self._open_item(again)
                    page = self._wait_page(again, timeout=12)
            if page is None:
                shot = self._save_shot("page-mismatch")
                self.log.error("点击后未能确认目标视频: %s，停止以避免继续点错（截图: %s）", found.title, shot or "无")
                raise FatalError("泛雅视频未切换", f"点击 {item.index} 后页面标题/当前目录项仍未匹配")
            if "视频" not in page.tabs:
                self.log.info("该任务点没有视频标签，跳过: %s", found.title)
                continue
            # 必须先切到视频标签再判断“任务点已完成”，否则可能在章节测验
            # 标签页上误读成视频已完成而跳过。
            if page.active_tab != "视频":
                self._click_tab(page, "视频")
                time.sleep(2.0)
                page = self._read()
            if page.video_completed:
                self._mark_video(found, "页面显示任务点已完成")
                self.log.info(
                    "视频任务点已完成（目录未打勾时以记录为准）: %s", found.title
                )
                continue
            self.watch_video(found)
            done += 1
            if self.mode == "auto" and self._practice_pending(item):
                if self._practice_item(item):
                    done_practice += 1
        videos = self.state.data.get("videos") if self.state is not None else {}
        practices = self.state.data.get("practices") if self.state is not None else {}
        self.log.info(
            "本次处理 %d 个视频、%d 个练习；累计视频 %d 条/练习 %d 条",
            done,
            done_practice,
            len(videos) if isinstance(videos, dict) else 0,
            len(practices) if isinstance(practices, dict) else 0,
        )
        return 0

    def _practice_item(self, item: FanyaItem) -> bool:
        self._check_stop()
        self.log.info(
            "下一个练习任务点: %s %s（状态=%s 计数=%s）",
            item.index, item.title, item.status, item.pending_count,
        )
        found = self.navigator.seek_forward(item.index) or self.navigator.seek_from_top(item.index)
        if found is None:
            self.log.warning("目录里找不到练习任务点 %s %s，跳过", item.index, item.title)
            return False
        if found.completed:
            self.log.info("练习目录项已是绿勾，跳过: %s %s", found.index, found.title)
            return False
        if not self._open_item(found):
            self.log.warning("练习目录项首次点击未生效，重试: %s", found.index)
            again = self.navigator.seek_from_top(found.index)
            if again is None or not self._open_item(again):
                self.log.warning("练习目录项无法点击: %s", item.title)
                return False
            found = again
        time.sleep(1.5)
        frame = self.screen.grab()
        # 只 OCR 页签区域，不再整页判断“像不像练习”。
        tab = self._find_text_rect(frame, "章节测验")
        if tab is not None:
            self.mouse.click_rect(tab.expand(8), "切换到章节测验")
            time.sleep(1.5)
        result = self.practice_solver.solve(self.layout)
        if result == "saved":
            self._mark_practice(found, "页面暂时保存成功")
            return True
        self.log.warning("练习未保存成功，保持原状: %s", item.title)
        return False

    def _find_text_rect(self, frame: np.ndarray, text: str) -> Optional[Rect]:
        if self.layout is None:
            return None
        content = self.layout.content
        top = Rect(content.x0, content.y0, content.x1, min(content.y1, int(0.35 * self.layout.screen_h)))
        try:
            lines = self.ocr.read(top.crop(frame), offset=(top.x0, top.y0), upscale=1.5)
        except Exception:  # noqa: BLE001
            return None
        for line in lines:
            if text in line.text:
                return line.rect
        return None

    def _save_shot(self, tag: str) -> Optional[str]:
        try:
            frame = self.screen.grab()
            out = self.cfg.path(str(self.cfg.runtime.shot_dir))
            out.mkdir(parents=True, exist_ok=True)
            path = out / time.strftime(f"fanya-{tag}-%Y%m%d-%H%M%S.png")
            if self.screen.save(path, frame):
                return str(path)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("保存截图失败: %s", exc)
        return None

    def _open_item(self, item: FanyaItem) -> bool:
        # 点击前重新读一次当前目录，使用最新坐标；扫描和点击之间目录可能
        # 因为自动定位/平滑滚动发生位移，用旧坐标会点到相邻条目。
        try:
            live_items = self.scanner.scan(self.screen.grab(), self.layout)
        except UserAbort:
            raise
        except Exception as exc:  # noqa: BLE001
            self.log.warning("点击前目录复扫失败: %s", exc)
            live_items = []
        fresh = next((it for it in live_items if it.index == item.index), None)
        if fresh is None:
            self.log.warning("点击前目录中看不到 %s，重新定位后再试", item.index)
            return False
        rect = fresh.title_rect
        panel = self.layout.panel if self.layout is not None else None
        if rect.width < 30 or rect.height < 8:
            self.log.error("目录项坐标异常，拒绝点击: %s %s", item.index, rect)
            return False
        if panel is not None and not (
            panel.x0 <= rect.cx <= panel.x1 and panel.y0 <= rect.cy <= panel.y1
        ):
            self.log.error("目录项坐标不在目录面板内，拒绝点击: %s %s", item.index, rect)
            return False
        try:
            self.mouse.click_rect(rect.expand(4), f"打开 {item.index}")
        except UserAbort:
            raise
        except Exception as exc:  # noqa: BLE001
            self.log.warning("点击目录项失败: %s", exc)
            return False
        time.sleep(2.5)
        return True

    def _click_tab(self, page: FanyaPageInfo, name: str) -> None:
        rect = page.tabs.get(name)
        if rect is None:
            return
        self.mouse.click_rect(rect.expand(8), f"切换到{name}")
        time.sleep(1.5)

    # ------------------------------------------------------------------ 视频监控
    def _wait_page(self, item: FanyaItem, timeout: float = 10) -> Optional[FanyaPageInfo]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._handle_prompt()
            frame = self.screen.grab()
            page = self.reader.read(frame, self.layout)
            if _title_matches(page.title, item.title):
                return page
            # 页面标题 OCR 有时只读到“返回课程”，此时用目录高亮行确认
            try:
                current = next(
                    (it for it in self.scanner.scan(frame, self.layout) if it.current), None
                )
            except Exception:  # noqa: BLE001
                current = None
            if current is not None and current.index == item.index:
                self.log.info("页面标题未读到，但目录当前项为 %s，确认已打开", item.index)
                return page
            time.sleep(1.0)
        return None

    def _check_stop(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise UserAbort("收到停止信号")

    def _show_controls(self, page: FanyaPageInfo) -> None:
        """轻微晃动鼠标唤出视频控制条，和智慧树版保持一致。"""
        enabled = getattr(
            getattr(self.cfg, "fanya", None),
            "controls_wiggle",
            getattr(self.cfg.player, "controls_wiggle", True),
        )
        if not bool(enabled):
            return
        x, y = page.video_rect.center
        try:
            self.mouse.move_to(x, y, duration=0.15)
            time.sleep(0.12)
            self.mouse.wiggle(x, y)
        except UserAbort:
            raise
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _paused_icon(frame: np.ndarray, rect: Optional[Rect]) -> Optional[bool]:
        """播放键是白色三角形=暂停中；两条竖线=正在播放；认不出返回 None。"""
        if rect is None:
            return None
        crop = frame[rect.y0 : rect.y1, rect.x0 : rect.x1]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = (gray > 190).astype(np.uint8)
        count, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        areas = [
            int(stats[i, cv2.CC_STAT_AREA])
            for i in range(1, count)
            if int(stats[i, cv2.CC_STAT_AREA]) >= 25
        ]
        if not areas:
            return None
        if len(areas) == 1 and areas[0] >= 80:
            return True
        if len(areas) >= 2:
            return False
        return None

    def _read(self) -> FanyaPageInfo:
        assert self.layout is not None
        return self.reader.read(self.screen.grab(), self.layout)

    def _ensure_playing(self, page: FanyaPageInfo) -> FanyaPageInfo:
        for attempt in range(4):
            self._show_controls(page)
            time.sleep(0.8)
            page = self._read()
            self.log.debug(
                "播放检测: 时间=%r 播放键=%s 状态=%s",
                page.time_text,
                page.play_button,
                page.current,
            )
            if page.video_completed:
                return page
            if page.current is not None:
                before = page.current
                time.sleep(2.2)
                # 控制条会自动隐藏；读数前先唤出，避免把“隐藏”误判为
                # 视频没动，随后点到底部播放键把正在播放的视频按停。
                self._show_controls(page)
                after = self._read()
                if after.current is not None and after.current > before + 0.4:
                    self.log.info("视频已开始播放: %s", after.time_text)
                    return after
            if self._read_video_quiz() is not None:
                # 开始播放后立即弹题时，继续点中央播放键会误点题目/选项。
                # 立即停手，交给 watch_video 的弹题处理。
                self.log.info("开始播放期间检测到视频弹题，暂停自动点播放")
                return page
            # 控制条会自动隐藏；判定播放/暂停前必须重新唤出，并只相信
            # 左侧图标的明确形状：三角形=暂停，双竖线=播放中，未知=不点。
            self._show_controls(page)
            time.sleep(0.35)
            frame = self.screen.grab()
            paused = self._paused_icon(frame, page.play_button)
            if paused is False:
                self.log.info("播放图标显示为播放中，停止自动点播放")
                return self._read()
            if page.play_button is not None and paused is True:
                self.mouse.click_rect(page.play_button, "点击播放")
                time.sleep(1.2)
                page = self._read()
                if page.current is not None and page.current > 0:
                    return page
                continue
            if page.current is None or page.current == 0:
                # 只有明确看到画面中央的大播放键时才点中央；否则未知状态
                # 点画面本身也会把正在播放的视频暂停。
                center = self._center_play_button(frame, page.video_rect)
                if center is not None:
                    self.log.info("新视频，点击画面中央播放键")
                    self.mouse.click(*center, "点击画面中央播放")
                    time.sleep(1.3)
                    page = self._read()
                    if page.current is not None and page.current > 0:
                        return page
                else:
                    self.log.debug("未识别到中央播放键，暂不点击画面")
            else:
                self.log.debug("播放图标状态未知，暂不点击（避免误暂停）")
            time.sleep(0.8)
        return page

    @staticmethod
    def _center_play_button(frame: np.ndarray, video_rect: Rect) -> Optional[tuple[int, int]]:
        """识别画面中央的大灰/白圆形播放键；普通画面不返回中心点。"""
        if video_rect.width < 80 or video_rect.height < 80:
            return None
        crop = frame[video_rect.y0:video_rect.y1, video_rect.x0:video_rect.x1]
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        # 只看中间区域，避免把底部控制条/字幕误当播放键
        region = crop[int(0.22 * h):int(0.78 * h), int(0.22 * w):int(0.78 * w)]
        if region.size == 0:
            return None
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(30, int(0.1 * region.shape[1])),
            param1=90,
            param2=28,
            minRadius=max(12, int(0.025 * h)),
            maxRadius=max(20, int(0.13 * h)),
        )
        if circles is None:
            return None
        best = min(
            circles[0],
            key=lambda c: (c[0] - region.shape[1] / 2) ** 2 + (c[1] - region.shape[0] / 2) ** 2,
        )
        cx, cy, radius = (float(v) for v in best[:3])
        if radius < 12:
            return None
        return (
            video_rect.x0 + int(0.22 * w + cx),
            video_rect.y0 + int(0.22 * h + cy),
        )

    def _handle_prompt(self) -> None:
        assert self.layout is not None
        frame = self.screen.grab()
        content = self.layout.content
        lines = self.ocr.read(content.crop(frame), offset=(content.x0, content.y0), upscale=1.1)
        text = "\n".join(ln.text for ln in lines)
        for word in STOP_WORDS:
            if word in text:
                raise FatalError("泛雅要求身份验证", f"页面出现“{word}”，脚本已停止，请手动处理")
        for word in PROMPT_WORDS:
            hit = next((ln for ln in lines if word in ln.text), None)
            if hit is not None:
                self.log.warning("检测到弹窗“%s”，点击继续", word)
                self.mouse.click_rect(hit.rect.expand(10), word)
                time.sleep(1.0)
                return

    def _read_video_quiz(self) -> Optional[VideoQuiz]:
        assert self.layout is not None
        content = self.layout.content
        # 提交/继续按钮在视频右下角，可能比正文区域更靠右（整卡 OCR 会裁掉）；
        # 用整屏右下区域再放大识别。
        region = Rect(
            int(0.62 * self.layout.screen_w),
            int(0.28 * self.layout.screen_h),
            self.layout.screen_w,
            int(0.92 * self.layout.screen_h),
        )
        try:
            return self.quiz_reader.read(self.screen.grab(), content, button_region=region)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("视频弹题识别失败: %s", exc)
            return None

    @staticmethod
    def _video_quiz_key(quiz: VideoQuiz) -> str:
        raw = quiz.kind + "|" + _norm(quiz.question) + "|" + "|".join(
            f"{k}.{_norm(v)}" for k, v in sorted(quiz.options.items())
        )
        return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()

    @staticmethod
    def _video_quiz_candidates(quiz: VideoQuiz, tried) -> list[tuple[str, ...]]:
        """按单选全选项、多选 1..N 组合的顺序生成未尝试答案。"""
        letters = sorted(quiz.options)
        used = {tuple(sorted(x)) for x in tried}
        out: list[tuple[str, ...]] = []
        if quiz.kind == "多选题":
            for size in range(1, len(letters) + 1):
                for combo in itertools.combinations(letters, size):
                    if tuple(sorted(combo)) not in used:
                        out.append(tuple(sorted(combo)))
        else:
            for letter in letters:
                if (letter,) not in used:
                    out.append((letter,))
        return out

    def _select_video_quiz_options(
        self, quiz: VideoQuiz, target: tuple[str, ...], attempt: dict
    ) -> None:
        target_set = set(target)
        current = set(attempt.get("selected") or [])
        if quiz.kind == "多选题":
            for letter in sorted(current - target_set):
                rect = quiz.option_rects.get(letter)
                if rect is not None:
                    self.mouse.click_rect(rect.expand(4), f"弹题取消 {letter}")
                    time.sleep(0.35)
            for letter in sorted(target_set - current):
                rect = quiz.option_rects.get(letter)
                if rect is not None:
                    self.mouse.click_rect(rect.expand(4), f"弹题选择 {letter}")
                    time.sleep(0.35)
        else:
            for letter in target:
                rect = quiz.option_rects.get(letter)
                if rect is not None:
                    self.mouse.click_rect(rect.expand(4), f"弹题选择 {letter}")
                    time.sleep(0.4)
        attempt["selected"] = list(target)

    def _solve_video_quiz(self, quiz: VideoQuiz, item: FanyaItem) -> None:
        if quiz.continue_button is not None:
            self.log.info("视频弹题已提交，点击继续")
            self.mouse.click_rect(quiz.continue_button, "弹题继续")
            time.sleep(1.0)
            return
        if not quiz.options:
            self.log.warning("检测到视频弹题，但暂未读到选项，等待下一次识别（不盲点提交）")
            return
        key = self._video_quiz_key(quiz)
        attempt = self._video_quiz_attempts.setdefault(
            key, {"tried": [], "selected": []}
        )
        self.log.info(
            "视频弹题: %s %s 选项=%s",
            quiz.kind,
            quiz.question[:80],
            quiz.options,
        )
        # 先使用大模型答案；失败后依次遍历其余选项/组合，绝不重复同一答案。
        if not attempt["tried"]:
            letters: list[str] = []
            if self.llm is not None and bool(getattr(self.cfg, "llm_ready", lambda: False)()):
                try:
                    frame = self.screen.grab()
                    question = QuizQuestion(
                        kind=quiz.kind,
                        text=quiz.question,
                        options=quiz.options,
                        option_rects=quiz.option_rects,
                        dialog=quiz.dialog,
                        needs_vision=True,
                    )
                    answer = self.llm.answer(
                        question, image_bgr=quiz.dialog.crop(frame), auto_switch=True
                    )
                    letters = [x for x in answer.letters if x in quiz.options]
                    self.log.info(
                        "视频弹题大模型答案: %s (%.2f) %s",
                        answer.letters,
                        answer.confidence,
                        answer.reason[:80],
                    )
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("视频弹题大模型失败: %s", exc)
            if not letters:
                letters = [sorted(quiz.options)[0]]
                self.log.warning("视频弹题无有效大模型答案，默认从 %s 开始尝试", letters[0])
            target = tuple(sorted(set(letters if quiz.kind == "多选题" else letters[:1])))
        else:
            candidates = self._video_quiz_candidates(quiz, attempt["tried"])
            if not candidates:
                self.log.error("视频弹题所有答案组合均已尝试仍错误，停止重试: %s", quiz.question[:60])
                return
            target = candidates[0]
        max_attempts = max(6, 1 + len(self._video_quiz_candidates(quiz, [])))
        for index in range(max_attempts):
            self._check_stop()
            if not attempt["tried"]:
                target = target
            else:
                candidates = self._video_quiz_candidates(quiz, attempt["tried"])
                if not candidates:
                    self.log.error("视频弹题答案组合用尽，停止: %s", quiz.question[:60])
                    return
                target = candidates[0]
            self.log.info("视频弹题尝试第 %d 次: %s", index + 1, list(target))
            self._select_video_quiz_options(quiz, tuple(target), attempt)
            if quiz.submit is not None:
                self.mouse.click_rect(quiz.submit, "弹题提交")
            else:
                self.log.warning("视频弹题没有读到提交按钮")
                return
            time.sleep(2.0)
            after = self._read_video_quiz()
            text = " ".join(line.text for line in (after.raw_lines if after else []))
            wrong = any(word in text for word in ("回答错误", "答错", "答案错误"))
            if self.state is not None:
                self.state.note_quiz(wrong=wrong)
            if after is not None and after.continue_button is not None:
                self.mouse.click_rect(after.continue_button, "弹题继续")
                self._video_quiz_attempts.pop(key, None)
                time.sleep(1.0)
                self.log.info("视频弹题回答正确，已继续")
                return
            attempt["tried"].append(tuple(target))
            if wrong:
                self.log.warning("视频弹题回答错误，切换下一个答案")
                # 错误提示后选项仍在；用最新识别结果进入下一轮
                if after is not None and after.options:
                    quiz = after
                time.sleep(0.6)
                continue
            # 没有继续、也没有明确错误：可能仍在处理，给平台一点时间
            time.sleep(1.5)
            after = self._read_video_quiz()
            text = " ".join(line.text for line in (after.raw_lines if after else []))
            if after is not None and after.continue_button is not None:
                self.mouse.click_rect(after.continue_button, "弹题继续")
                self._video_quiz_attempts.pop(key, None)
                time.sleep(1.0)
                self.log.info("视频弹题已继续")
                return
            if any(word in text for word in ("回答错误", "答错", "答案错误")):
                self.log.warning("视频弹题回答错误，切换下一个答案")
                continue
            self.log.warning("视频弹题提交后状态未知，停止自动重试，等待下一轮识别")
            return
        self.log.error("视频弹题重试次数用尽，仍未继续: %s", quiz.question[:60])

    def watch_video(self, item: FanyaItem) -> None:
        quiz = self._read_video_quiz()
        if quiz is not None:
            self._solve_video_quiz(quiz, item)
        page = self._ensure_playing(self._read())
        quiz = self._read_video_quiz()
        if quiz is not None:
            self._solve_video_quiz(quiz, item)
            page = self._read()
        if page.video_completed:
            self._mark_video(item, "页面显示任务点已完成")
            self.log.info("视频任务点已标记完成: %s", item.title)
            return
        fcfg = getattr(self.cfg, "fanya", None)
        progress_timeout = float(
            getattr(fcfg, "progress_timeout_seconds", 15.0) or 15.0
        )
        max_recoveries = int(getattr(fcfg, "max_recoveries", 4) or 4)
        wiggle_interval = float(
            getattr(fcfg, "controls_wiggle_interval_seconds", 10.0) or 10.0
        )
        last_controls = time.monotonic()
        last_current = page.current
        last_advance = time.monotonic()
        last_heartbeat = 0.0
        recoveries = 0
        self.log.info("开始监控视频: %s (%s)", item.title, page.time_text)
        quiz_count = 0
        while True:
            self._check_stop()
            time.sleep(2.0)
            # 先看视频弹题：题目里可能出现“登录/验证”等词，不能被页面级
            # 停止词误判；弹题本身也不属于“继续学习”公告。
            quiz = self._read_video_quiz()
            if quiz is not None:
                quiz_count += 1
                if quiz_count > 8:
                    self.log.error("视频弹题反复出现，停止当前视频: %s", item.title)
                    return
                self._solve_video_quiz(quiz, item)
                last_current = None
                last_advance = time.monotonic()
                continue
            self._handle_prompt()
            page = self._read()
            now = time.monotonic()
            if page.video_completed:
                self._mark_video(item, "页面显示任务点已完成")
                self.log.info("任务点已完成: %s", item.title)
                return
            if page.title and not _title_matches(page.title, item.title):
                self.log.info("页面已切换到其他内容，停止监控: %s", page.title)
                return
            if page.current is None or now - last_controls > wiggle_interval:
                self._show_controls(page)
                page = self._read()
                last_controls = time.monotonic()
                now = last_controls
            if page.current is not None:
                if last_current is None or page.current > last_current + 0.4:
                    last_current = page.current
                    last_advance = now
                if page.total and page.current >= page.total - 2:
                    self.log.info("播放到达结尾，等待平台标记完成: %s", page.time_text)
                    if self._wait_completed():
                        self._mark_video(item, "播放结束且平台标记完成")
                        self.log.info("任务点已完成: %s", item.title)
                    else:
                        self.log.warning("到达结尾但平台未标记完成: %s", item.title)
                    return
            if now - last_advance > progress_timeout:
                self._show_controls(page)
                time.sleep(0.35)
                page = self._read()
                last_controls = time.monotonic()
                now = last_controls
                if (
                    page.current is not None
                    and (last_current is None or page.current > last_current + 0.4)
                ):
                    last_current = page.current
                    last_advance = now
                    continue
                frame = self.screen.grab()
                paused = self._paused_icon(frame, page.play_button)
                if paused is False:
                    # 双竖线说明视频仍在播放（可能缓冲/控制条未刷新），
                    # 不应按“暂停”去点播放键，更不能因此跳过任务。
                    self.log.debug("图标显示正在播放，按缓冲处理，不计无进度")
                    last_advance = time.monotonic()
                    continue
                recoveries += 1
                if recoveries > max_recoveries:
                    self.log.error("视频长时间无进度，跳过当前任务点: %s", item.title)
                    return
                self.log.warning(
                    "视频 %ds 无进度（时间=%r 播放键=%s 图标=%s），尝试恢复 (%d/%d)",
                    int(now - last_advance),
                    page.time_text,
                    page.play_button,
                    paused,
                    recoveries,
                    max_recoveries,
                )
                if page.play_button is not None and paused is True:
                    self.mouse.click_rect(page.play_button, "恢复播放")
                else:
                    # 观看中途不能用中央圆形检测兜底：视频内容里的圆形
                    # （图表、人物等）会被误判成播放键，点一下反而暂停。
                    self.log.debug("播放状态未知，暂不点击（避免误暂停）")
                last_advance = time.monotonic()
            if now - last_heartbeat > 30:
                last_heartbeat = now
                self.log.info(
                    "观看中: %s %s (%s)",
                    item.index,
                    page.time_text or "?",
                    f"{page.ratio:.1%}" if page.ratio is not None else "?",
                )

    def _wait_completed(self) -> bool:
        deadline = time.monotonic() + COMPLETION_GRACE
        while time.monotonic() < deadline:
            self._handle_prompt()
            page = self._read()
            if page.video_completed:
                return True
            time.sleep(1.5)
        return False
