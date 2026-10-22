from __future__ import annotations

import logging
import re
import threading
import time

import cv2
from pathlib import Path
from typing import Optional

import numpy as np

from . import vision
from .config import Config
from .layout import Layout
from .llm import LLMClient, LLMError
from .mouse import InputUnavailable, Mouse
from .notify import Notifier
from .ocr import OcrEngine, OcrUnavailable
from .panel import PanelNavigator, PanelScanner
from .practice import PracticeEntry, PracticePage, PracticeScanner
from .player import PlayerMonitor, _fmt
from .quiz import QuizHandler
from .screen import Screen, ScreenUnavailable
from .state import State
from .types import (
    STATUS_NOT_STARTED,
    FatalError,
    Lesson,
    LessonError,
    PlaybackState,
    Rect,
    UserAbort,
    normalize_title,
)

log = logging.getLogger("autolesson.controller")

LOGIN_WORDS = ("扫码登录", "账号登录", "密码登录", "登录后", "请登录", "重新登录")
VERIFY_WORDS = (
    "人脸识别",
    "人脸验证",
    "请完成验证",
    "验证码",
    "拖动滑块",
    "滑块验证",
    "请正对摄像头",
    "实名认证",
)
NETWORK_WORDS = ("网络异常", "网络错误", "网络中断", "连接失败", "加载失败", "重新加载", "请求失败")
QUIZ_WORDS = ("弹题", "单选题", "多选题", "判断题")
MUST_READ_WORDS = ("感谢你选择修读", "学习指导卡")
DAILY_WORDS = ("每日6点更新", "每日6點更新", "当前时间计划学习进度")
# 只匹配 AI 助教的固定界面文案，不能匹配学生姓名。
_AI_ICON_PATH = Path(__file__).with_name("assets") / "ai_assistant_icon.png"
_AI_ICON_CACHE = None
AI_ASSISTANT_WORDS = (
    "AI助教",
    "AI 助教",
    "AI助教小智",
    "我是您的AI助教",
    "欢迎随时提问",
    "每日6点更新",
    "每日6點更新",
    "当前时间计划学习进度",
    "计划学习进度",
)
GENERIC_WORDS = ("确定", "知道了", "我知道了", "继续学习")


def merge_pending_items(lessons, tests, is_pending, done_tests) -> list[tuple]:
    """按目录 y 坐标合并未完成视频与未完成平时测试，保持列表顺序。"""
    items = [(l.title_rect.cy, 0, "video", l) for l in lessons if is_pending(l)]
    items += [
        (t.rect.cy, 1, "test", t)
        for t in tests
        if not t.completed and t.key not in done_tests
    ]
    items.sort(key=lambda x: (x[0], x[1]))
    return items


class AutoLesson:
    def __init__(
        self,
        cfg: Config,
        ocr: OcrEngine,
        screen: Screen,
        mouse: Mouse,
        state: State,
        notifier: Notifier,
        logger: Optional[logging.Logger] = None,
        stop_event: Optional[threading.Event] = None,
        list_mode: bool = False,
    ):
        self.cfg = cfg
        self.ocr = ocr
        self.screen = screen
        self.mouse = mouse
        self.state = state
        self.notify = notifier
        self.log = logger or log
        self.stop_event = stop_event or threading.Event()
        self.llm = LLMClient(cfg, state, self.log.getChild("llm")) if cfg.quiz.enabled else None
        self.quiz = (
            QuizHandler(
                ocr, cfg, mouse, screen, self.llm, state, self.log.getChild("quiz"),
                notifier=self.notify,
            )
            if cfg.quiz.enabled
            else None
        )
        self.player = PlayerMonitor(ocr, cfg, self.log.getChild("player"))
        self.layout: Optional[Layout] = None
        self.navigator: Optional[PanelNavigator] = None
        self._last_frame: Optional[np.ndarray] = None
        self._solving_quiz = False
        self._completed_this_run = 0
        self._started_at = time.monotonic()
        # 本次运行中曾经看到过“已完成”的视频；用于抵抗滚动瞬间的状态漏识别
        self._known_completed: set[str] = set()
        self.list_mode = bool(list_mode)
        self.practice_scanner = PracticeScanner(ocr, cfg)
        self._tests_done: set[str] = set()
        self._test_attempts: dict[str, int] = {}

    # ================================================================== 入口
    def run(self) -> int:
        try:
            self._run()
            return 0
        except UserAbort as exc:
            self.log.warning("用户中止: %s", exc)
            self.state.save()
            shot = self._save_shot("abort")
            self.notify.send("AutoLesson 已中止", str(exc), "warning", shot, force=True)
            return 130
        except FatalError as exc:
            shot = self._save_shot("fatal")
            self.log.error("致命故障: %s | %s", exc.reason, exc.detail)
            self.state.save()
            self.notify.send(
                "AutoLesson 已停止",
                f"{exc.reason}\n{exc.detail}\n\n{self.state.summary()}",
                "fatal",
                shot,
                force=True,
            )
            return 2
        except (OcrUnavailable, ScreenUnavailable, InputUnavailable) as exc:
            shot = self._save_shot("fatal")
            self.notify.send("AutoLesson 无法启动", str(exc), "fatal", shot, force=True)
            return 2
        except KeyboardInterrupt:
            self.log.warning("收到 Ctrl+C，已停止")
            self.state.save()
            return 130
        except Exception as exc:  # noqa: BLE001
            self.log.exception("未预期异常")
            shot = self._save_shot("fatal")
            self.notify.send("AutoLesson 崩溃", f"{type(exc).__name__}: {exc}", "fatal", shot, force=True)
            return 2
        finally:
            self.state.save()
            try:
                self.screen.close()
            except Exception:  # noqa: BLE001
                pass

    def _run(self) -> None:
        self.state.note_run()
        self.log.info("AutoLesson 启动，配置: %s", self.cfg.source_path or "内置默认值")
        frame = self._grab("启动")
        self.layout = self._calibrate(frame)
        self.scanner = PanelScanner(self.ocr, self.cfg)
        self.navigator = PanelNavigator(
            self.scanner, self.ocr, self.cfg, self.mouse, self.screen, self.layout
        )
        # AI 助教弹窗会遮挡目录/状态图标，必须先收起再处理课程页
        self._prepare_ai_assistant()
        # 再处理启动时可能存在的“学前必读/每日进度/弹题”，然后检查课程页面
        if self._handle_overlays(frame):
            frame = self._grab("关闭启动弹窗")
            if not self.layout.calibrated:
                lines = self.ocr.read(frame)
                self.layout = Layout.detect(frame, lines, self.cfg)
                self.scanner = PanelScanner(self.ocr, self.cfg)
                self.navigator = PanelNavigator(
                    self.scanner, self.ocr, self.cfg, self.mouse, self.screen, self.layout
                )
        self._check_course_page(frame)
        if not self.layout.calibrated:
            frame = self._grab("重新标定")
            lines = self.ocr.read(frame)
            refreshed = Layout.detect(frame, lines, self.cfg)
            if refreshed.calibrated:
                self.layout = refreshed
                self.scanner = PanelScanner(self.ocr, self.cfg)
                self.navigator = PanelNavigator(
                    self.scanner, self.ocr, self.cfg, self.mouse, self.screen, self.layout
                )
                self.log.info("页面加载完成后已重新标定目录面板")
        self._prepare_ai_assistant()
        self.notify.send(
            "AutoLesson 开始挂机",
            f"课程页面已就绪，屏幕 {self.layout.screen_w}x{self.layout.screen_h}，"
            "脚本将自动观看未完成视频。请勿移动鼠标或锁屏。\n"
            f"按 {self.cfg.runtime.stop_hotkey} 可随时安全停止。",
            "info",
        )

        all_lessons = sorted(self.navigator.scan_all(), key=self._lesson_order)
        if not self.navigator.last_scan_complete:
            raise FatalError(
                "目录扫描不完整",
                f"翻到扫描上限仍未到列表底部，只读到 {len(all_lessons)} 个视频。"
                "请检查滚轮是否能滚动目录，或调大 panel.max_scrolls；"
                "为避免选错视频，本次不开始观看。",
            )
        self._expected_lesson_count = len(all_lessons)
        if not all_lessons:
            raise FatalError(
                "目录中没有视频",
                "右侧目录未解析出任何视频。请确认章节已展开、页面已加载，或运行 calibrate 检查布局。",
            )
        for lesson in all_lessons:
            if lesson.completed:
                self._remember_completed(lesson)
        pending = [l for l in all_lessons if self._is_pending(l)]
        self.log.info(
            "目录共 %d 个视频，已完成 %d，待学习 %d",
            len(all_lessons),
            len(all_lessons) - len(pending),
            len(pending),
        )
        if not pending and not self.list_mode:
            self._finish_success()
            return

        self.navigator.scroll_to_top()
        without_progress = 0
        last_pending_key: tuple[str, ...] = ()
        while not self.stop_event.is_set():
            self._check_stop()
            self._check_caps()
            frame = self._grab("主循环")
            if self._handle_overlays(frame):
                continue
            lessons = self.navigator.scan(frame)
            tests = (
                self.practice_scanner.scan(frame, self.layout)
                if self.list_mode
                else []
            )
            if not lessons and not tests:
                self._recover_no_lessons()
                continue
            for lesson in lessons:
                if lesson.completed:
                    self._remember_completed(lesson)
            items = (
                merge_pending_items(lessons, tests, self._is_pending, self._tests_done)
                if self.list_mode
                else [(l.title_rect.cy, 0, "video", l) for l in lessons if self._is_pending(l)]
            )
            if not items:
                moved = self.navigator.scroll_down()
                if moved:
                    continue
                # 已到列表底部，做一次全量复核
                self.log.info("当前页无未完成视频/题目，执行全量复核")
                all_lessons = self.navigator.scan_all()
                if not self.navigator.last_scan_complete:
                    raise FatalError(
                        "目录复核不完整",
                        f"复核时仍未滚到列表底部，只读到 {len(all_lessons)} 个视频。"
                        "为避免漏看或选错，已停止。",
                    )
                for lesson in all_lessons:
                    if lesson.completed:
                        self._remember_completed(lesson)
                frame = self._grab("复核")
                tests = (
                    self.practice_scanner.scan(frame, self.layout)
                    if self.list_mode
                    else []
                )
                remaining = [l for l in all_lessons if self._is_pending(l)]
                remaining_tests = [
                    t for t in tests if not t.completed and t.key not in self._tests_done
                ]
                keys = tuple(
                    sorted([l.key for l in remaining] + [t.key for t in remaining_tests])
                )
                if not remaining and not remaining_tests:
                    if self._expected_lesson_count and len(all_lessons) < self._expected_lesson_count:
                        raise FatalError(
                            "目录复核不完整",
                            f"首次扫描到 {self._expected_lesson_count} 个视频，复核只读到 {len(all_lessons)} 个。"
                            "可能页面未加载完，已停止以避免漏看。",
                        )
                    self._finish_success()
                    return
                if keys == last_pending_key:
                    without_progress += 1
                    if without_progress >= 2:
                        names = "、".join(
                            f"{l.index or ''}{l.title}" for l in remaining[:8]
                        ) or "、".join(t.title for t in remaining_tests[:8])
                        raise FatalError("仍有视频/题目无法完成", f"未完成: {names}")
                else:
                    without_progress = 0
                    last_pending_key = keys
                self.navigator.scroll_to_top()
                continue

            _y, _kind_order, kind, item = items[0]
            if kind == "test":
                self.log.info("下一个平时测试: %s", item)
                self._process_test(item)
                without_progress = 0
                continue
            candidate = item
            self.log.info("下一个视频: %s", candidate)
            self._process_with_retry(candidate)
            self._completed_this_run += 1
            without_progress = 0
        raise UserAbort("收到停止信号")

    @staticmethod
    def _lesson_order(lesson: Lesson) -> tuple:
        """按章节号排序（2.2 在 5.8 前），避免扫描插入顺序导致跳着选。"""
        text = (getattr(lesson, "index", "") or getattr(lesson, "title", "") or "").strip()
        m = re.match(r"(\d+)\s*[.．]\s*(\d+)", text)
        if m:
            return (int(m.group(1)), int(m.group(2)), text)
        return (999, 999, text)

    # ================================================================== 布局/检查
    def _calibrate(self, frame: np.ndarray) -> Layout:
        if vision.is_black(frame):
            raise FatalError("屏幕全黑", "可能已锁屏、息屏或远程桌面断开")
        lines = self.ocr.read(frame)
        layout = Layout.detect(frame, lines, self.cfg)
        labels = [ln.text for ln in lines if "目录" in ln.text.replace(" ", "")]
        if not labels:
            self.log.warning("未识别到“目录”二字，使用比例布局，仍会继续尝试")
        return layout

    def _check_course_page(self, frame: np.ndarray) -> None:
        for attempt in range(4):
            lines = self.ocr.read(frame)
            panel_lines = [ln for ln in lines if ln.rect.cx > self.layout.panel_left]
            has_catalog = any("目录" in ln.text.replace(" ", "") for ln in panel_lines)
            has_duration = any(re.search(r"\d{1,2}:\d{2}:\d{2}", ln.text) for ln in panel_lines)
            if has_catalog or has_duration:
                return
            critical = self._critical_text(frame)
            if critical:
                self._raise_critical(critical)
            self.log.warning("尚未识别到课程目录（第 %d/4 次），等待页面加载", attempt + 1)
            time.sleep(3)
            frame = self._grab("页面检查")
        raise FatalError(
            "未识别到课程页面",
            "请确认浏览器已打开智慧树课程学习页（studyvideoh5.zhihuishu.com/stuStudy），"
            "并保持窗口可见、页面已登录。",
        )

    def _recover_no_lessons(self) -> None:
        self.log.warning("目录为空，检查页面状态")
        frame = self._grab("目录为空")
        if self._handle_overlays(frame):
            return
        critical = self._critical_text(frame)
        if critical:
            self._raise_critical(critical)
        time.sleep(3)
        for _ in range(5):
            self._check_stop()
            if self.navigator and self.navigator.scan():
                return
            time.sleep(2)
        raise FatalError("无法读取课程目录", "页面可能仍在加载、被弹窗遮挡或已跳转")

    # ================================================================== 状态缓存/重试
    def _remember_completed(self, lesson: Lesson) -> None:
        self._known_completed.add(lesson.key)
        self._known_completed.add(lesson.title_key)

    def _is_known_completed(self, lesson: Lesson) -> bool:
        return lesson.key in self._known_completed or lesson.title_key in self._known_completed

    def _is_pending(self, lesson: Lesson) -> bool:
        if lesson.completed:
            return False
        # 本地完成记录（state.json 的 lessons）优先跳过，减少重复扫描后再次点击
        if self.state.is_completed(lesson):
            return False
        # 本轮完整扫描已确认完成的视频，后续视口即使因为滚动/边缘裁切把它
        # 瞬时读成“未学习/学习中”，也不再重复播放。此前 2.7 已完成后又被
        # 读成 in_progress，导致选错视频，就是这里需要以完成缓存为准。
        if self._is_known_completed(lesson):
            return False
        return True

    # ================================================================== 单个平时测试
    def _process_test(self, entry: PracticeEntry) -> None:
        """在列表顺序模式里处理一个未完成的平时测试，只暂存不提交。"""
        if self.stop_event.is_set():
            return
        if self.state.is_practice_done(entry.key):
            self.log.info("记录显示该平时测试已暂存/提交，跳过: %s", entry)
            self._tests_done.add(entry.key)
            return
        self.log.info("打开平时测试（列表顺序）: %s", entry)
        self.mouse.click_rect(entry.rect, "平时测试")
        time.sleep(float(getattr(self.cfg.practice, "open_wait", 2.0)))
        page = PracticePage(
            self.ocr,
            self.cfg,
            self.mouse,
            self.screen,
            self.llm,
            self.state,
            self.log.getChild("practice"),
            self.stop_event,
            notifier=self.notify,
        )
        try:
            result = page.run(close_tab=True)
        except Exception as exc:  # noqa: BLE001
            self.log.exception("平时测试处理失败: %s", exc)
            page.close_tab()
            result = {"opened": False, "error": str(exc)}
        if result.get("not_ready"):
            n = self._test_attempts.get(entry.key, 0) + 1
            self._test_attempts[entry.key] = n
            if n < 2:
                self.log.warning("平时测试页未就绪，稍后重试: %s", entry)
                return
            self.log.error("平时测试页重试后仍未就绪，本次跳过: %s", entry)
        self._tests_done.add(entry.key)
        if result.get("saved") or result.get("submitted"):
            self.state.mark_practice(entry.key, entry.chapter, entry.title)
        self.state.save()
        self.log.info("平时测试结果: %s", result)

    # ================================================================== 单个视频
    def _process_with_retry(self, lesson: Lesson) -> Lesson:
        """失败后立即重试同一个视频，而不是跳到列表里的下一个。"""
        attempts = max(1, int(self.cfg.runtime.max_lesson_attempts))
        current = lesson
        for attempt in range(1, attempts + 1):
            try:
                self._process_lesson(current)
                return current
            except LessonError as exc:
                self.log.error("处理视频失败(%d/%d): %s | %s", attempt, attempts, current, exc)
                shot = self._save_shot("lesson-fail")
                if attempt >= attempts:
                    raise FatalError(
                        "视频重复失败",
                        f"{current.index or ''} {current.title} 已连续失败 {attempts} 次，"
                        "停止以避免影响课程",
                    ) from exc
                self.notify.send(
                    "视频处理失败，准备重试同一个视频",
                    f"{current.index or ''} {current.title}\n{exc}\n失败次数: {attempt}/{attempts}",
                    "warning",
                    shot,
                )
                self._sleep(3)
                found = self._find_lesson_anywhere(current)
                if found is None:
                    raise FatalError("重试时找不到视频", f"目录中找不到: {current.title}")
                if found.completed:
                    self.log.info("重试前发现平台已标记完成: %s", found)
                    self.state.mark_completed(found)
                    self._remember_completed(found)
                    return found
                current = found
        return current

    def _process_lesson(self, lesson: Lesson) -> None:
        self.log.info("开始处理: %s", lesson)
        started = time.monotonic()
        if not self._click_lesson(lesson):
            self.log.info("点击前平台已标记完成，跳过观看: %s", lesson)
            self.state.mark_completed(lesson)
            self._remember_completed(lesson)
            return
        self._sleep(float(self.cfg.window.settle_seconds))
        frame = self._grab("进入视频")
        self._handle_overlays(frame)
        self._verify_loaded(lesson)
        self._ensure_playing()
        self._watch_video(lesson)
        self._sleep(float(self.cfg.player.completion_grace_seconds))

        if not self._verify_completed(lesson):
            self.log.warning("首次播放结束后平台未标记完成，重试一次: %s", lesson)
            self.navigator.scroll_to_top()
            found = self._find_lesson_anywhere(lesson)
            if found is None:
                raise LessonError("播放结束后在目录中找不到该视频")
            if not self._click_lesson(found):
                self.state.mark_completed(found)
                self._remember_completed(found)
                return
            self._sleep(float(self.cfg.window.settle_seconds))
            self._handle_overlays(self._grab("重试进入"))
            self._ensure_playing()
            self._watch_video(found)
            self._sleep(float(self.cfg.player.completion_grace_seconds))
            if not self._verify_completed(found):
                raise LessonError("视频已播放完，但平台始终未标记完成")
            lesson = found

        self.state.mark_completed(lesson)
        self._remember_completed(lesson)
        self.state.note_watch(time.monotonic() - started)
        self.log.info("完成: %s | %s", lesson, self.state.summary())

    def _click_lesson(self, lesson: Lesson) -> bool:
        """点击目标视频；若点击前发现平台已标记完成，返回 False。"""
        for attempt in range(3):
            self._check_stop()
            frame = self._grab("点击视频前")
            if self._handle_overlays(frame):
                frame = self._grab("关闭弹窗后")
            current = self.navigator.scan(frame)
            target = PanelNavigator.find(current, lesson)
            if target is None:
                self.log.warning("目标视频不在当前视图，重新滚动查找")
                target = self._find_lesson_anywhere(lesson)
                if target is None:
                    raise LessonError(f"目录中找不到视频: {lesson.title}")
                frame = self._grab("定位视频后")
                current = self.navigator.scan(frame)
                target = PanelNavigator.find(current, lesson) or target
            if target.completed:
                self.log.info("点击前发现平台已标记完成，跳过: %s", target)
                self._remember_completed(target)
                return False
            screen_h = self.layout.screen_h
            if target.title_rect.y0 < 8 or target.title_rect.y1 > screen_h - 8:
                self.log.info("目标视频在屏幕外，先向上滚动")
                self.mouse.scroll_at(self.layout.panel.cx, self.layout.panel.cy, 4)
                time.sleep(0.8)
                continue
            if target.title_rect.y0 < self.layout.panel_top + 4:
                # panel_top 可能比实际可见列表头更低；只要行还完整显示在屏幕上就直接点
                self.log.info("目标视频在目录顶部附近但仍可见，直接点击")
            self.log.info("点击视频: %s @ %s", target, target.title_rect.center)
            self.mouse.click_rect(target.title_rect, f"打开视频 {target.index or ''}")
            time.sleep(2.0)
            frame = self._grab("点击视频后")
            text = self._breadcrumb_text(frame)
            if self._breadcrumb_matches(text, lesson):
                return True
            self.log.warning("面包屑不匹配: %r, 目标 %s，重试点击", text, lesson)
            time.sleep(1.0)
        raise LessonError(f"点击后未切换到目标视频: {lesson.title}")

    def _verify_loaded(self, lesson: Lesson) -> None:
        # 播放器控制条会自动隐藏，且点击画面会切换暂停/控件；
        # 只把鼠标移到控制条区域唤出，不再点击画面，并给慢网络更充裕的时间。
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            frame = self._grab("验证播放器")
            if self._handle_overlays(frame):
                continue
            state = self.player.refresh(self.screen, self.layout, self.mouse, attempts=2)
            if state is not None:
                if state.total and lesson.duration:
                    diff = abs(state.total - lesson.duration)
                    if diff > float(self.cfg.player.expected_total_tolerance):
                        self.log.warning(
                            "视频总时长 %s 与目录预计 %s 相差 %ds，可能进入的不是目标视频",
                            _fmt(state.total),
                            _fmt(lesson.duration),
                            diff,
                        )
                self.log.info("播放器已加载: %s", state.raw_time)
                return
            self.player.show_controls(self.mouse, self.layout)
            time.sleep(1.0)
        raise LessonError("视频播放器未加载出进度条")

    def _click_play_once(self, state: Optional[PlaybackState] = None) -> bool:
        """只点一次播放按钮。

        不再先点画面：画面点击是暂停/播放切换，和播放按钮连点会把刚开始的视频
        又按停。这里只把鼠标移到左下控制条附近唤出控件，然后点一次播放键。
        """
        if self._handle_overlays(self._grab("点击播放前确认")):
            return False
        if state is None or state.play_button is None:
            state = self.player.refresh(self.screen, self.layout, self.mouse, attempts=2)
        if state is None or state.play_button is None:
            return False
        # 双保险：播放键必须在控制条最左侧；否则可能是“下一节”按钮
        if state.play_button.cx > int(self.layout.player_width * 0.04):
            self.log.warning(
                "播放按钮位置异常 x=%d（可能是下一节按钮），跳过点击",
                state.play_button.cx,
            )
            return False
        # 只把鼠标移到视频中央（不点击），让控件出现后再点播放键。
        # 不再移到左下角，避免鼠标指针看起来停在播放器左下角。
        x, y = self.layout.player_center
        self.mouse.move_to(x, y, duration=0.15)
        time.sleep(0.25)
        self.player.click_play(self.mouse, state, "播放(单次)")
        return True

    def _click_video_center(self) -> None:
        """点击视频正中央，避开左下控制条/播放键/下一节。"""
        x, y = self.layout.player_center
        self.mouse.click(x, y, "点击视频中央")

    def _ensure_playing(self) -> PlaybackState:
        video_clicked = False
        for attempt in range(6):
            self._check_stop()
            frame = self._grab("确保播放")
            if self._handle_overlays(frame):
                continue
            state = self.player.refresh(self.screen, self.layout, self.mouse, attempts=2)
            if state is None:
                # 只移动鼠标到控制条附近唤出控件，不点击画面
                x, y = self.layout.player_center
                self.mouse.move_to(x, y, duration=0.15)
                time.sleep(0.3)
                state = self.player.refresh(self.screen, self.layout, self.mouse, attempts=2)
            if state is None:
                # 控件没出现时也先点视频正中央，而不是控制条/进度条区域
                if not video_clicked and attempt >= 2:
                    if self._handle_overlays(self._grab("点击视频前确认")):
                        continue
                    self._click_video_center()
                    video_clicked = True
                    time.sleep(1.0)
                time.sleep(1.0)
                continue

            # 以“时间/进度是否前进”为准，不盲信播放图标。
            # 播放三角和暂停双竖线在局部对比度掩码下形状很接近，容易互相误判。
            advanced = self.player.wait_for_advance(
                self.screen, self.layout, state.current, timeout=5, mouse=self.mouse
            )
            if advanced is not None and self._advanced(state, advanced):
                self.log.info("视频已开始播放: %s", advanced.raw_time or advanced.percent)
                return advanced

            # 优先点视频正中央（不是进度条/控制条），再验证是否前进。
            if not video_clicked:
                if self._handle_overlays(self._grab("点击视频前确认")):
                    continue
                self._click_video_center()
                video_clicked = True
                time.sleep(1.0)
                advanced = self.player.wait_for_advance(
                    self.screen, self.layout, state.current, timeout=5, mouse=self.mouse
                )
                if advanced is not None and self._advanced(state, advanced):
                    self.log.info("视频已开始播放: %s", advanced.raw_time or advanced.percent)
                    return advanced

            # 中央点击失败后，只点控制条最左侧的播放键；下一节键在它右边，必须排除。
            self.log.info("进度未前进（图标判定=%s），点击左下角播放按钮", state.playing)
            if self._click_play_once(None):
                advanced = self.player.wait_for_advance(
                    self.screen, self.layout, state.current, timeout=5, mouse=self.mouse
                )
                if advanced is not None and self._advanced(state, advanced):
                    self.log.info("视频已开始播放: %s", advanced.raw_time or advanced.percent)
                    return advanced
            self.log.debug("第 %d 次尝试后仍未检测到进度变化", attempt + 1)
            time.sleep(1.0)
        raise LessonError("视频无法开始播放（已尝试点击视频中央与左下角播放键）")

    @staticmethod
    def _advanced(before: PlaybackState, after: PlaybackState) -> bool:
        if before.current is not None and after.current is not None:
            return after.current > before.current + 0.4
        if before.ratio is not None and after.ratio is not None:
            return after.ratio > before.ratio + 0.0005
        return False

    # ================================================================== 观看监控
    def _watch_video(self, lesson: Lesson) -> None:
        poll = float(self.cfg.player.poll_seconds)
        pause_grace = float(self.cfg.player.pause_grace_seconds)
        stall_timeout = float(self.cfg.player.stall_timeout_seconds)
        no_bar_timeout = float(self.cfg.player.no_bar_timeout_seconds)
        max_recoveries = int(self.cfg.player.max_recoveries)
        expected_total = float(lesson.duration)
        last_real_advance = time.monotonic()
        last_play_click = 0.0
        last_bar = time.monotonic()
        last_current: Optional[float] = None
        last_ratio: Optional[float] = None
        recoveries = 0
        last_heartbeat = 0.0
        last_breadcrumb = time.monotonic()
        resume_needed = False
        self.log.info("开始监控播放: %s (%s)", lesson.title, lesson.duration_text())

        while True:
            self._check_stop()
            now = time.monotonic()
            frame = self._grab("观看中")
            if vision.is_black(frame):
                raise FatalError("屏幕全黑", "可能锁屏或显示器休眠")
            if self._handle_overlays(frame):
                last_real_advance = time.monotonic()
                last_play_click = time.monotonic()
                last_current = None
                last_ratio = None
                resume_needed = True
                continue

            if resume_needed:
                resume_needed = False
                self._resume_after_overlay()
                continue

            state = self.player.snapshot(frame, self.layout)
            if state is None:
                idle_no_bar = now - last_bar
                if idle_no_bar > no_bar_timeout:
                    text = self._breadcrumb_text(frame)
                    if not self._breadcrumb_matches(text, lesson):
                        self.log.warning("页面已切换，停止监控当前视频")
                        return
                    raise FatalError(
                        "长时间找不到播放进度条",
                        f"{idle_no_bar:.0f}s 内控件始终未出现，页面可能异常",
                    )
                if bool(self.cfg.player.controls_wiggle):
                    x, y = self.layout.player_center
                    self.mouse.wiggle(x, y)
                self._sleep(1.0)
                continue

            last_bar = now
            total = state.total or expected_total
            current = state.current
            ratio = state.ratio
            if current is not None:
                position = current
            elif ratio is not None and total:
                position = ratio * total
            else:
                position = None

            if current is not None and last_current is not None and current < last_current - 5:
                self.log.warning("播放位置回退（%.0f -> %.0f），可能已切换/循环", last_current, current)
                return
            if ratio is not None and last_ratio is not None and ratio < last_ratio - 0.02:
                self.log.warning("进度大幅回退，可能已切换/循环")
                return

            completed = False
            if current is not None and total:
                completed = current >= total - 1.5
            if ratio is not None and ratio >= float(self.cfg.player.completion_ratio):
                completed = True
            if completed:
                self.log.info("播放到达结尾: %s", state.raw_time or state.percent)
                return

            if current is not None:
                if last_current is None or current > last_current + 0.4:
                    last_current = current
                    last_real_advance = now
            if ratio is not None:
                if last_ratio is None or ratio > last_ratio + 0.0005:
                    last_ratio = ratio
                    last_real_advance = now

            idle = now - last_real_advance
            if idle > stall_timeout:
                self.log.warning("%ds 无播放进展，执行恢复 (%d/%d)", int(idle), recoveries + 1, max_recoveries)
                if recoveries >= max_recoveries:
                    self._handle_stall_failure(lesson, frame)
                    # 只有网络恢复时 _handle_stall_failure 才会返回，重置计数继续观看
                    recoveries = 0
                    last_real_advance = time.monotonic()
                    last_play_click = time.monotonic()
                    continue
                recoveries += 1
                self._try_recover_playback(state)
                last_play_click = time.monotonic()
            elif idle > pause_grace and now - last_play_click > pause_grace:
                # 图标可能把暂停误判成播放；只要时间/进度不走，就点一次播放按钮。
                # 先刷新让控件出现，立即点击，避免等几秒后控件已自动隐藏。
                self.log.info("检测到 %ds 无进度（图标=%s），先点视频中央", int(idle), state.playing)
                x, y = self.player.video_center(self.layout)
                self.mouse.click(x, y, "点击视频中央恢复播放")
                time.sleep(0.8)
                resumed = self.player.wait_for_advance(
                    self.screen, self.layout, state.current, timeout=2.5, mouse=self.mouse
                )
                if resumed is None or not self._advanced(state, resumed):
                    # 中央点击没恢复，再点左侧播放键
                    self.log.info("视频中央点击后仍未前进，点击左下播放按钮")
                    if not self._click_play_once(state):
                        self.mouse.click(x, y, "点击视频中央恢复播放")
                last_play_click = time.monotonic()

            if now - last_breadcrumb > float(self.cfg.player.breadcrumb_check_seconds):
                last_breadcrumb = now
                text = self._breadcrumb_text(frame)
                if not self._breadcrumb_matches(text, lesson):
                    self.log.warning("面包屑显示已切换视频，停止监控当前视频: %r", text)
                    return

            if now - last_heartbeat > 30:
                last_heartbeat = now
                pct = f"{state.percent:.1f}%" if state.percent is not None else "?"
                self.log.info(
                    "观看中: %s %s/%s (%s) 状态=%s",
                    lesson.index or "",
                    state.raw_time or "?",
                    _fmt(total) if total else "?",
                    pct,
                    {True: "播放", False: "暂停", None: "未知"}[state.playing],
                )
            self._sleep(poll)

    def _resume_after_overlay(self) -> None:
        """答题/必读弹窗关闭后检查是否恢复，未恢复只点一次播放键。"""
        self._sleep(1.5)
        frame = self._grab("弹窗关闭后")
        if self._handle_overlays(frame):
            return
        state = self.player.refresh(self.screen, self.layout, self.mouse, attempts=2)
        if state is None:
            self.log.info("弹窗关闭后仍未找到控件，移动到控制条并点一次播放")
            self._click_play_once(None)
            return
        advanced = self.player.wait_for_advance(
            self.screen, self.layout, state.current, timeout=3, mouse=self.mouse
        )
        if advanced is not None and self._advanced(state, advanced):
            self.log.info("弹窗关闭后视频已自动恢复")
            return
        self.log.info("弹窗关闭后视频未恢复，先点视频中央")
        x, y = self.player.video_center(self.layout)
        self.mouse.click(x, y, "点击视频中央恢复播放")
        time.sleep(0.8)
        resumed = self.player.wait_for_advance(
            self.screen, self.layout, state.current, timeout=2.5, mouse=self.mouse
        )
        if resumed is not None and self._advanced(state, resumed):
            return
        self.log.info("视频中央点击后仍未前进，点击左下播放按钮")
        self._click_play_once(state)

    def _try_recover_playback(self, state: PlaybackState) -> None:
        # 卡顿时只点一次播放，避免“点画面 + 点播放键”两次切换造成暂停/播放循环
        self.log.info("卡顿恢复：点击一次播放")
        self._click_play_once(state)
        x, y = self.player.video_center(self.layout)
        self.mouse.wiggle(x, y)
        self._sleep(3.0)

    def _handle_stall_failure(self, lesson: Lesson, frame: np.ndarray) -> None:
        frame = self._grab("卡顿检查")
        critical = self._critical_text(frame)
        if critical:
            self._raise_critical(critical)
        network = self._find_network_error(frame)
        if network:
            self.log.warning("检测到网络提示 %r，等待恢复", network)
            if self._wait_network(lesson):
                return
            raise FatalError("网络中断且无法恢复", f"等待 {int(self.cfg.player.network_wait_seconds)}s 后仍无进展")
        self._last_frame = frame
        self._save_shot("stalled")
        raise FatalError(
            "播放长时间无进展",
            f"{lesson.index or ''} {lesson.title} 已尝试 {self.cfg.player.max_recoveries} 次恢复\n"
            f"最后状态: {frame.shape[1]}x{frame.shape[0]}",
        )

    def _wait_network(self, lesson: Lesson) -> bool:
        deadline = time.monotonic() + float(self.cfg.player.network_wait_seconds)
        while time.monotonic() < deadline:
            self._check_stop()
            self._sleep(5)
            frame = self._grab("等待网络")
            if self._handle_overlays(frame):
                continue
            state = self.player.refresh(self.screen, self.layout, self.mouse, attempts=1)
            if state is not None and state.playing is False:
                self.player.click_play(self.mouse, state)
            if state is not None and state.current is not None:
                time.sleep(3)
                again = self.player.refresh(self.screen, self.layout, self.mouse, attempts=1)
                if again is not None and again.current is not None and again.current > state.current + 0.5:
                    self.log.info("网络恢复，继续播放")
                    return True
        return False

    # ================================================================== 完成校验
    def _verify_completed(self, lesson: Lesson) -> bool:
        for attempt in range(3):
            self._sleep(2.0 if attempt == 0 else 5.0)
            frame = self._grab("校验完成")
            if self._handle_overlays(frame):
                continue
            current = self.navigator.scan(frame)
            found = PanelNavigator.find(current, lesson)
            if found is not None and found.completed:
                self.log.info("平台已标记完成: %s", found)
                self._remember_completed(found)
                return True
            if found is not None and found.status != "completed":
                self.log.info("平台暂未标记完成(%s)，等待刷新", found.status)
            # 不在当前视图：从顶部完整查找
            self.navigator.scroll_to_top()
            for _ in range(int(self.cfg.panel.max_scrolls)):
                frame = self._grab("查找视频")
                current = self.navigator.scan(frame)
                found = PanelNavigator.find(current, lesson)
                if found is not None:
                    if found.completed:
                        self._remember_completed(found)
                        return True
                    self.log.info("目录状态仍为 %s，等待", found.status)
                    break
                if not self.navigator.scroll_down():
                    break
        return False

    def _find_lesson_anywhere(self, lesson: Lesson) -> Optional[Lesson]:
        self.navigator.scroll_to_top()
        for _ in range(int(self.cfg.panel.max_scrolls)):
            self._check_stop()
            frame = self._grab("查找视频")
            current = self.navigator.scan(frame)
            found = PanelNavigator.find(current, lesson)
            if found is not None:
                return found
            if not self.navigator.scroll_down():
                break
        return None

    # ================================================================== AI 助教
    @staticmethod
    def _ai_text(lines: list) -> str:
        return "\n".join(
            line.text for line in lines
            if any(word in line.text for word in AI_ASSISTANT_WORDS)
        )

    @staticmethod
    def _find_x_in(lines: list, rect: Rect):
        from .quiz import find_x_button

        found = find_x_button(lines, rect)
        if found is not None:
            return found
        candidates = [
            line for line in lines
            if line.text.strip() in ("X", "x", "×", "✕", "╳")
            and rect.contains(line.cx, line.cy)
        ]
        if not candidates:
            return None
        # 弹窗/悬浮卡的关闭按钮在右上角：优先最靠右、最靠上
        line = max(candidates, key=lambda ln: (ln.cx, -ln.cy))
        return line.rect.expand(8)

    @staticmethod
    def _ai_icon_template():
        global _AI_ICON_CACHE
        if _AI_ICON_CACHE is None:
            img = cv2.imread(str(_AI_ICON_PATH), cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                _AI_ICON_CACHE = (img[:, :, :3], img[:, :, 3])
            elif img is not None:
                _AI_ICON_CACHE = (img, None)
            else:
                _AI_ICON_CACHE = (None, None)
        return _AI_ICON_CACHE

    def _find_ai_icon(self, frame: np.ndarray) -> Optional[Rect]:
        """用 AI 悬浮球原图做多尺度模板匹配，不依赖文字。"""
        template, alpha = self._ai_icon_template()
        if template is None:
            return None
        panel = self.layout.panel
        x0 = max(0, panel.x0 - 100)
        x1 = min(self.layout.screen_w, panel.x1)
        y1 = min(self.layout.screen_h, panel.y1)
        search = frame[0:y1, x0:x1]
        if search.size == 0:
            return None
        template_h, template_w = template.shape[:2]
        best = None
        for width in range(80, 241, 20):
            height = max(40, int(round(template_h * width / template_w)))
            if width >= search.shape[1] or height >= search.shape[0]:
                continue
            tpl = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
            mask = None
            if alpha is not None:
                mask = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST)
            try:
                if mask is not None:
                    result = cv2.matchTemplate(search, tpl, cv2.TM_SQDIFF_NORMED, mask=mask)
                else:
                    result = cv2.matchTemplate(search, tpl, cv2.TM_SQDIFF_NORMED)
            except cv2.error:
                continue
            result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
            min_v, _max_v, min_loc, _max_loc = cv2.minMaxLoc(result)
            if min_v > 0.12:
                continue
            if best is None or min_v < best[0]:
                best = (
                    min_v,
                    Rect(x0 + min_loc[0], min_loc[1], x0 + min_loc[0] + width, min_loc[1] + height),
                )
        if best is None:
            return None
        rect = best[1]
        crop = frame[rect.y0 : rect.y1, rect.x0 : rect.x1]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        colored = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 80)).mean()
        if colored < 0.08:
            self.log.info("AI 悬浮球候选颜色不足(%.3f)，判定为误匹配: %s", colored, rect)
            return None
        self.log.info("AI 助教悬浮球匹配: 差异 %.3f 彩色 %.3f %s", best[0], colored, rect)
        return rect

    def _prepare_ai_assistant(self) -> None:
        """收起 AI 弹窗；悬浮球直接拖到目录上方，不再转成标签。"""
        if not bool(self.cfg.get("overlay.ai_assistant_cleanup", True)):
            self.log.info("AI 助教清理已关闭（overlay.ai_assistant_cleanup=false）")
            return

        # 1) 大弹窗会盖住悬浮球，先按固定界面文案点 X 收起
        for _ in range(3):
            frame = self._grab("检查AI助教")
            lines = self.ocr.read(frame)
            ai_lines = [
                line for line in lines
                if any(word in line.text for word in AI_ASSISTANT_WORDS)
            ]
            if not ai_lines:
                break
            popup = ai_lines[0].rect
            for line in ai_lines[1:]:
                popup = popup.union(line.rect)
            popup = popup.expand(80).clamp(self.layout.screen_w, self.layout.screen_h)
            button = self._find_x_in(lines, popup)
            if button is None:
                self.log.info("检测到 AI 助教弹窗，但未找到关闭按钮")
                break
            self.log.info("收起 AI 助教弹窗: %s", button.center)
            self.mouse.click_rect(button, "收起AI助教弹窗")
            self._sleep(1.2)

        # 2) 确认弹窗已经消失，避免把弹窗里的头像误当成悬浮球
        frame = self._grab("确认AI助教弹窗")
        lines = self.ocr.read(frame)
        if any(any(word in line.text for word in AI_ASSISTANT_WORDS) for line in lines):
            self.log.info("AI 助教弹窗仍未消失，跳过悬浮球拖动以免误点")
            return

        # 3) 模板匹配悬浮球；没有就明确跳过，不猜、不乱点
        icon = self._find_ai_icon(frame)
        if icon is None:
            self.log.info("未检测到 AI 悬浮球，跳过（该课程页面可能没有悬浮球）")
            return

        panel = self.layout.panel
        if icon.y1 <= panel.y0 - 8:
            self.log.info("AI 悬浮球已在目录上方，无需移动: %s", icon.center)
            return

        target_center = max(icon.height // 2 + 12, panel.y0 - icon.height // 2 - 24)
        if icon.cy <= target_center + 30:
            self.log.info("AI 悬浮球已经靠上，无需移动: %s", icon.center)
            return

        # 4) 直接拖动悬浮球到目录上方；用户已验证悬浮球本身可拖动
        for attempt in range(2):
            self.log.info(
                "拖动 AI 悬浮球到目录上方(第%d次): %s -> y=%d",
                attempt + 1,
                icon.center,
                target_center,
            )
            self.mouse.drag(icon.cx, icon.cy, icon.cx, target_center, "拖动AI悬浮球")
            self._sleep(0.9)
            frame = self._grab("检查AI悬浮球位置")
            moved = self._find_ai_icon(frame)
            if moved is None:
                self.log.info("拖动后未再检测到 AI 悬浮球（可能已吸附/隐藏）")
                return
            if moved.y1 <= panel.y0 - 8 or moved.cy <= target_center + 30:
                self.log.info("AI 悬浮球拖动完成: %s", moved.center)
                return
            icon = moved
            target_center = max(icon.height // 2 + 12, panel.y0 - icon.height // 2 - 24)
        self.log.warning("AI 悬浮球拖动后仍在下方的 %s，请人工检查", icon.center)

    # ================================================================== 弹窗
    def _handle_overlays(self, frame: np.ndarray) -> bool:
        if self.layout is None or self._solving_quiz:
            return False
        cards = vision.find_cards(
            frame,
            self.layout.panel_left,
            int(self.cfg.overlay.card_threshold),
            int(self.cfg.overlay.card_close_kernel),
            int(self.cfg.overlay.max_cards),
        )
        for card in cards:
            if card.width < 0.18 * self.layout.screen_w or card.height < 0.12 * self.layout.screen_h:
                continue
            # 顶部工具条会铺满播放器宽度，视频画面也可能铺满；真正的弹窗会留有边距
            if (
                card.x0 <= 2
                and card.x1 >= self.layout.panel_left - 2
                and card.y0 < 0.20 * self.layout.screen_h
            ):
                continue
            if card.y0 <= 2 and card.y1 >= self.layout.screen_h - 2:
                continue
            # 白底 PPT/视频画面通常铺满播放器宽度；真正的弹窗更窄，
            # 先按尺寸排除，避免每个轮询都对整张幻灯片做两次 OCR。
            if (
                card.width >= 0.85 * self.layout.player_width
                and card.height >= 0.35 * self.layout.screen_h
                and (card.x0 <= 2 or card.x1 >= self.layout.player_width - 2)
            ):
                continue
            header_rect = Rect(card.x0, card.y0, card.x1, min(card.y1, card.y0 + max(90, int(0.28 * card.height))))
            footer_rect = Rect(card.x0, max(card.y0, card.y1 - max(60, int(0.16 * card.height))), card.x1, card.y1)
            header = self.ocr.read_text(header_rect.crop(frame), offset=(header_rect.x0, header_rect.y0), upscale=1.3)
            footer = self.ocr.read_text(footer_rect.crop(frame), offset=(footer_rect.x0, footer_rect.y0), upscale=1.3)
            text = header + "\n" + footer
            kind = self._classify_overlay(text)
            if kind is None:
                self.log.debug("忽略非弹窗白色卡片 %s (%.0f%% 屏宽): %r", card, 100 * card.width / self.layout.screen_w, header[:40])
                continue
            # 弹题弹窗位于播放器中央，宽度明显小于播放器；白色 PPT 页可能含“单选题”等字样
            if kind == "quiz" and (
                card.width > 0.70 * self.layout.player_width
                or not (0.25 * self.layout.player_width < card.cx < 0.80 * self.layout.player_width)
                or card.height > 0.88 * self.layout.screen_h
            ):
                self.log.debug("疑似视频画面而非弹题，忽略: %s", card)
                continue
            if kind == "generic" and card.width > 0.75 * self.layout.player_width:
                self.log.debug("疑似视频画面而非提示框，忽略: %s", card)
                continue
            self.log.info("检测到弹窗[%s] %s: %r", kind, card, header[:60].replace("\n", " "))
            if kind == "quiz":
                self._solve_quiz(frame, card)
                return True
            if kind in ("login", "verify"):
                self._raise_critical(kind)
            if kind == "network":
                if self._wait_network_for_overlay(card):
                    return True
                raise FatalError("网络错误弹窗无法消除", text[:200])
            self._close_overlay(card, text, kind)
            return True
        return False

    def _solve_quiz(self, frame: np.ndarray, card: Rect) -> None:
        if self.quiz is None:
            raise FatalError("遇到弹题但答题功能已禁用", "quiz.enabled=false")
        self._solving_quiz = True
        try:
            result = self.quiz.solve(frame, card)
            self.log.info("弹题处理完成: %s", result)
        except LLMError as exc:
            self._last_frame = self._grab("答题失败")
            raise FatalError("大模型答题失败", str(exc)) from exc
        except Exception:
            self._last_frame = self._grab("答题失败")
            raise
        finally:
            self._solving_quiz = False

    @staticmethod
    def _classify_overlay(text: str) -> Optional[str]:
        t = (text or "").replace(" ", "")
        if not t:
            return None
        if any(word in t for word in QUIZ_WORDS) and (
            "弹题" in t or "【" in t or "关闭" in t or "正确答案" in t
        ):
            return "quiz"
        if any(word in t for word in MUST_READ_WORDS):
            return "must_read"
        if any(word in t for word in DAILY_WORDS) or ("同学" in t and "学习进度" in t):
            return "daily"
        if any(word in t for word in LOGIN_WORDS):
            return "login"
        if any(word in t for word in VERIFY_WORDS):
            return "verify"
        if any(word in t for word in NETWORK_WORDS):
            return "network"
        if any(word in t for word in GENERIC_WORDS):
            return "generic"
        return None

    def _close_overlay(self, card: Rect, text: str, kind: str = "generic") -> None:
        frame = self._grab("关闭弹窗")
        lines = self.ocr.read(card.crop(frame), offset=(card.x0, card.y0), upscale=1.2)
        from .quiz import find_close_button, find_x_button

        button = find_close_button(lines, card) or find_x_button(lines, card)
        if button is not None:
            self.mouse.click_rect(button, "关闭弹窗")
        else:
            self.mouse.click(card.x1 - int(0.035 * card.width), card.y0 + int(0.05 * card.height), "关闭弹窗(X)")
        self._sleep(1.2)
        if not self._card_still_there(card, kind):
            return
        # 尝试“确定/知道了”
        frame = self._grab("弹窗仍在")
        lines = self.ocr.read(card.crop(frame), offset=(card.x0, card.y0), upscale=1.2)
        for line in lines:
            if any(word in line.text for word in ("确定", "知道了", "我知道了", "继续学习")):
                self.mouse.click_rect(line.rect.expand(6), "确认弹窗")
                self._sleep(1.0)
                if not self._card_still_there(card, kind):
                    return
                break
        self.mouse.press("esc")
        self._sleep(1.0)
        if not self._card_still_there(card, kind):
            return
        raise FatalError("弹窗无法关闭", f"内容: {text[:200]}")

    def _card_still_there(self, card: Rect, kind: str = "generic") -> bool:
        frame = self._grab("检查弹窗")
        cards = vision.find_cards(
            frame,
            self.layout.panel_left,
            int(self.cfg.overlay.card_threshold),
            int(self.cfg.overlay.card_close_kernel),
            4,
        )
        for other in cards:
            if other.iou(card) > 0.45 or (other.area > 0.5 * card.area and other.contains(card.cx, card.cy)):
                # 关闭后如果露出白色 PPT，也会被当成卡片，必须用文字确认仍是同一个弹窗
                text = self.ocr.read_text(other.crop(frame), upscale=1.0)
                if kind == "generic":
                    if any(word in text for word in ("确定", "知道了", "继续学习", "关闭")):
                        return True
                elif self._classify_overlay(text) == kind:
                    return True
        return False

    def _wait_network_for_overlay(self, card: Rect) -> bool:
        self._sleep(5)
        return not self._card_still_there(card, "network")

    # ================================================================== 页面文字
    def _breadcrumb_text(self, frame: np.ndarray) -> str:
        if self.layout is None:
            return ""
        h = self.layout.screen_h
        # 只取蓝色标题栏下方、课程目录上方的面包屑，避开浏览器地址栏
        region = Rect(0, int(0.105 * h), int(0.55 * self.layout.screen_w), int(0.19 * h)).clamp(
            self.layout.screen_w, h
        )
        lines = self.ocr.read(region.crop(frame), offset=(region.x0, region.y0), upscale=1.2)
        return " ".join(ln.text for ln in lines)

    @staticmethod
    def _breadcrumb_matches(text: str, lesson: Lesson) -> bool:
        compact = normalize_title(text)
        if not compact:
            return True
        if lesson.index:
            indexes = re.findall(r"\d{1,2}\.\d{1,2}", compact)
            if indexes:
                return lesson.index in indexes
        title = normalize_title(lesson.title)
        if title:
            prefix = title[: min(6, len(title))]
            if prefix and prefix in compact:
                return True
            if compact in title:
                return True
        return True

    def _critical_text(self, frame: np.ndarray) -> Optional[str]:
        h, w = frame.shape[:2]
        region = Rect(0, 0, w, int(0.6 * h))
        text = self.ocr.read_text(region.crop(frame), upscale=0.75)
        for kind, words in (("verify", VERIFY_WORDS), ("login", LOGIN_WORDS), ("network", NETWORK_WORDS)):
            for word in words:
                if word in text:
                    return kind
        return None

    def _find_network_error(self, frame: np.ndarray) -> Optional[str]:
        h, w = frame.shape[:2]
        right = min(int(0.9 * w), self.layout.panel_left) if self.layout else int(0.9 * w)
        region = Rect(int(0.1 * w), int(0.1 * h), right, int(0.7 * h))
        text = self.ocr.read_text(region.crop(frame), upscale=0.8)
        for word in NETWORK_WORDS:
            if word in text:
                return word
        return None

    def _raise_critical(self, kind: str) -> None:
        if kind == "login":
            raise FatalError("登录状态失效", "页面要求重新登录，请手动登录后重新运行")
        if kind == "verify":
            raise FatalError("平台要求身份验证", "检测到人脸/验证码/滑块等验证，脚本立即停止，请手动处理")
        if kind == "network":
            raise FatalError("网络异常", "页面提示网络错误，请检查网络后重新运行")
        raise FatalError("页面异常", kind)

    # ================================================================== 杂项
    def _grab(self, tag: str = "") -> np.ndarray:
        frame = self.screen.grab()
        self._last_frame = frame
        if tag:
            self.log.debug("截图: %s %dx%d", tag, frame.shape[1], frame.shape[0])
        return frame

    def _save_shot(self, tag: str) -> Optional[Path]:
        if self._last_frame is None:
            return None
        name = time.strftime(f"{tag}-%Y%m%d-%H%M%S.png")
        path = self.cfg.path(str(self.cfg.runtime.shot_dir)) / name
        if self.screen.save(path, self._last_frame):
            return path
        return None

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            if self.stop_event.is_set():
                raise UserAbort("收到停止信号")
            time.sleep(min(0.25, max(0.0, end - time.monotonic())))

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise UserAbort("收到停止信号")

    def _check_caps(self) -> None:
        max_lessons = int(self.cfg.runtime.max_lessons)
        if max_lessons > 0 and self._completed_this_run >= max_lessons:
            raise UserAbort(f"已达到本次运行上限 {max_lessons} 个视频")
        max_minutes = float(self.cfg.runtime.max_minutes)
        if max_minutes > 0 and (time.monotonic() - self._started_at) / 60.0 >= max_minutes:
            raise UserAbort(f"已达到本次运行时间上限 {max_minutes:.0f} 分钟")

    def _finish_success(self) -> None:
        self.state.save()
        self.log.info("全部视频已完成。%s", self.state.summary())
        self.notify.send(
            "AutoLesson 全部完成",
            f"本课程所有视频已观看完成。本次完成 {self._completed_this_run} 节。\n{self.state.summary()}",
            "success",
            force=True,
        )
