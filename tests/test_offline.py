from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolesson import vision  # noqa: E402
from autolesson.config import load_config  # noqa: E402
from autolesson.controller import AutoLesson, merge_pending_items  # noqa: E402
from autolesson.hotkey import HotkeyListener, parse_hotkey  # noqa: E402
from autolesson.layout import Layout  # noqa: E402
from autolesson.llm import LLMAnswer, LLMClient, LLMError  # noqa: E402
from autolesson.ocr import OcrEngine  # noqa: E402
from autolesson.panel import PanelNavigator, PanelScanner, parse_duration  # noqa: E402
from autolesson.practice import PracticeEntry, PracticePage, PracticeRunner, PracticeScanner  # noqa: E402
from autolesson.player import PlayerMonitor, parse_clock, parse_time_range  # noqa: E402
from autolesson.quiz import QuizHandler, find_close_button, is_feedback_text, parse_feedback, parse_quiz  # noqa: E402
from autolesson.state import State  # noqa: E402
from autolesson.types import Lesson, LessonError, OcrLine, PlaybackState, QuizQuestion, Rect  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
RUNTIME = ROOT / "runtime"
RUNTIME.mkdir(exist_ok=True)


def load(name: str) -> np.ndarray:
    image = cv2.imread(str(FIXTURES / name))
    if image is None:
        raise unittest.SkipTest(f"缺少测试图 {name}")
    return image


def dummy_question() -> QuizQuestion:
    return QuizQuestion(
        kind="多选题",
        text="测试题",
        options={"A": "甲", "B": "乙", "C": "丙"},
        option_rects={},
        dialog=Rect(0, 0, 100, 100),
    )


class ParserTests(unittest.TestCase):
    def test_clock(self):
        self.assertEqual(parse_clock("11:56"), 716)
        self.assertEqual(parse_clock("1:02:03"), 3723)

    def test_time_range(self):
        self.assertEqual(parse_time_range("00:00:15/00:11:56"), (15, 716))
        self.assertEqual(parse_time_range("00:00:15 ／ 00:11:56"), (15, 716))

    def test_duration(self):
        self.assertEqual(parse_duration("● 00:19:36"), 1176)
        self.assertEqual(parse_duration("03:33"), 213)
        self.assertEqual(parse_duration("● 03:33"), 213)
        self.assertIsNone(parse_duration("平时测试"))

    def test_replica_mmss_and_course_title(self):
        scanner = PanelScanner(None, load_config())
        layout = Layout.manual(500, 300, panel_left=0, panel_right=500, panel_top=0)
        lines = [
            OcrLine("第1章第1课", Rect(80, 20, 180, 40)),
            OcrLine("1.1", Rect(10, 35, 40, 55)),
            OcrLine("03:33", Rect(80, 55, 130, 75)),
        ]
        lessons = scanner.parse(lines, np.zeros((300, 500, 3), np.uint8), layout)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0].title, "第1章第1课")
        self.assertEqual(lessons[0].duration, 213)

    def test_short_title_left_of_duration_is_kept(self):
        scanner = PanelScanner(None, load_config())
        layout = Layout.manual(500, 200, panel_left=0, panel_right=500, panel_top=0)
        lines = [
            OcrLine("1.1", Rect(10, 40, 40, 60)),
            OcrLine("短标题", Rect(60, 20, 140, 40)),
            OcrLine("00:12:30", Rect(400, 40, 460, 60)),
        ]
        lessons = scanner.parse(lines, np.zeros((200, 500, 3), np.uint8), layout)
        self.assertEqual(len(lessons), 1)
        self.assertEqual(lessons[0].title, "短标题")

    def test_feedback(self):
        self.assertEqual(parse_feedback("正确答案: C"), (["C"], True))
        self.assertEqual(parse_feedback("回答错误 正确答案：B"), (["B"], False))
        self.assertTrue(is_feedback_text("正确答案: C"))
        self.assertTrue(is_feedback_text("正确"))
        self.assertTrue(is_feedback_text("回答错误 正确答案：B"))
        self.assertFalse(is_feedback_text("以下哪项是正确答案？"))
        self.assertFalse(is_feedback_text("以下哪项回答正确？"))

    def test_llm_json(self):
        letters, confidence, _reason = LLMClient._parse(
            '{"answers":["A","C"],"confidence":0.8,"reason":"ok"}', dummy_question()
        )
        self.assertEqual(letters, ["A", "C"])
        self.assertAlmostEqual(confidence, 0.8)

    def test_llm_switches_image_to_text_on_empty(self):
        calls = []

        class FakeLLM(LLMClient):
            def __init__(self):
                self.log = logging.getLogger("test.llm.switch")

            def _answer_once(self, question, image_bgr=None):
                calls.append(image_bgr is not None)
                if len(calls) == 1:
                    raise LLMError("大模型返回无法解析: 空内容")
                return LLMAnswer(letters=["A"], confidence=0.5, reason="fallback", source="text")

        answer = FakeLLM().answer(dummy_question(), image_bgr=np.zeros((10, 10, 3), np.uint8))
        self.assertEqual(calls, [True, False])
        self.assertEqual(answer.letters, ["A"])

    def test_llm_switches_text_to_image_on_empty(self):
        calls = []

        class FakeLLM(LLMClient):
            def __init__(self):
                self.log = logging.getLogger("test.llm.switch")

            def _answer_once(self, question, image_bgr=None):
                calls.append(image_bgr is not None)
                if len(calls) == 1:
                    raise LLMError("大模型返回无法解析: 空内容")
                return LLMAnswer(letters=["B"], confidence=0.5, reason="fallback", source="image")

        answer = FakeLLM().answer(
            dummy_question(),
            image_bgr=None,
            alternate_image_bgr=np.zeros((10, 10, 3), np.uint8),
        )
        self.assertEqual(calls, [False, True])
        self.assertEqual(answer.letters, ["B"])

    def test_llm_fallback(self):
        self.assertEqual(LLMClient._parse("答案：B", dummy_question())[0], ["B"])

    def test_llm_verbose_answer_field(self):
        self.assertEqual(
            LLMClient._parse('{"answers":"Answer: C","confidence":0.8}', dummy_question())[0], ["C"]
        )

    def test_overlay_classifier(self):
        self.assertEqual(AutoLesson._classify_overlay("弹题测验 关闭"), "quiz")
        self.assertEqual(AutoLesson._classify_overlay("学前必读 感谢你选择修读"), "must_read")
        self.assertEqual(AutoLesson._classify_overlay("每日6点更新"), "daily")
        self.assertEqual(AutoLesson._classify_overlay("扫码登录"), "login")
        self.assertEqual(AutoLesson._classify_overlay("人脸识别"), "verify")

    def test_breadcrumb_matching(self):
        lesson = Lesson(
            index="1.2",
            title="行列式基础",
            duration=1176,
            status="in_progress",
            title_rect=Rect(0, 0, 10, 10),
            row=Rect(0, 0, 10, 10),
        )
        self.assertTrue(AutoLesson._breadcrumb_matches("第一章1.2、行列式基础", lesson))
        self.assertFalse(AutoLesson._breadcrumb_matches("第二章2.1、矩阵的定义", lesson))
        self.assertTrue(AutoLesson._breadcrumb_matches("", lesson))

    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            path = Path(tmp) / "state.json"
            state = State(path)
            lesson = Lesson(
                index="1.1",
                title="课程简介",
                duration=557,
                status="completed",
                title_rect=Rect(0, 0, 10, 10),
                row=Rect(0, 0, 10, 10),
            )
            state.mark_completed(lesson)
            state.cache_answer("abc", ["C"], 0.9, "测试题")
            reloaded = State(path)
            self.assertTrue(reloaded.is_completed(lesson))
            self.assertEqual(reloaded.cached_answer("abc")["letters"], ["C"])

    def test_config_env(self):
        cfg = load_config()
        self.assertGreater(cfg.player.poll_seconds, 0)
        self.assertEqual(cfg.layout.panel_left, 0)

    def test_api_key_resolution(self):
        cfg = load_config()
        # 配置里的 api_key 是字符串，不能被同名方法遮蔽
        self.assertEqual(cfg.llm.api_key, "")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            self.assertEqual(cfg.resolved_api_key(), "sk-test")

    def test_deepseek_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in ("LLM_BASE_URL", "LLM_MODEL", "LLM_VISION_MODEL", "DEEPSEEK_API_KEY"):
                os.environ.pop(key, None)
            cfg = load_config()
        self.assertEqual(cfg.llm.base_url, "https://api.deepseek.com")
        self.assertEqual(cfg.llm.model, "deepseek-flash")
        self.assertEqual(cfg.llm.vision_model, "")
        self.assertTrue(cfg.llm.json_mode)
        self.assertFalse(cfg.llm.thinking)
        self.assertEqual(cfg.llm.api_key_env, "DEEPSEEK_API_KEY")


class FakeResponse:
    def __init__(self, status_code=200, content='{"answers":["C"],"confidence":0.9,"reason":"ok"}', text=None):
        self.status_code = status_code
        self._content = content
        self.text = content if text is None else text

    def json(self):
        return {
            "choices": [{"message": {"content": self._content}}],
            "usage": {"total_tokens": 12},
        }


class LLMPayloadTests(unittest.TestCase):
    def _client(self):
        tmp = tempfile.TemporaryDirectory(dir=RUNTIME)
        self.addCleanup(tmp.cleanup)
        cfg = load_config()
        cfg.llm.api_key = "sk-test"
        cfg.llm.cache = False
        cfg.llm.vision_model = ""
        cfg.llm.json_mode = True
        cfg.llm.thinking = False
        cfg.llm.max_retries = 0
        state = State(Path(tmp.name) / "state.json")
        return LLMClient(cfg, state)

    @staticmethod
    def _fake_post(calls, responses):
        def fake(url, headers=None, data=None, timeout=None):
            payload = json.loads(data)
            calls.append(payload)
            response = responses.pop(0)
            if callable(response):
                return response(payload)
            return response

        return fake

    def test_text_payload_uses_deepseek_json_mode(self):
        calls = []
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            client = self._client()
            with patch("requests.Session.post", side_effect=self._fake_post(calls, [FakeResponse()])):
                answer = client.answer(dummy_question())
        self.assertEqual(answer.letters, ["C"])
        payload = calls[0]
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 400)

    def test_image_uses_model_when_vision_model_empty(self):
        calls = []
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            client = self._client()
            question = QuizQuestion(
                kind="单选题",
                text="下列行列式的值为",
                options={"A": "-42", "B": "42", "C": "0"},
                option_rects={},
                dialog=Rect(0, 0, 100, 100),
            )
            with patch("requests.Session.post", side_effect=self._fake_post(calls, [FakeResponse()])):
                answer = client.answer(question, image_bgr=np.full((80, 120, 3), 255, np.uint8))
        self.assertEqual(answer.source, "image")
        payload = calls[0]
        self.assertEqual(payload["model"], "deepseek-flash")
        content = payload["messages"][1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_api_400_strips_unknown_extras(self):
        calls = []
        responses = [
            FakeResponse(400, text="unknown field thinking"),
            FakeResponse(200, '{"answers":["C"],"confidence":0.9}'),
        ]
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            client = self._client()
            with patch("requests.Session.post", side_effect=self._fake_post(calls, responses)):
                answer = client.answer(dummy_question())
        self.assertEqual(answer.letters, ["C"])
        self.assertEqual(len(calls), 2)
        self.assertIn("thinking", calls[0])
        self.assertNotIn("thinking", calls[1])
        self.assertNotIn("response_format", calls[1])

    def test_empty_json_content_retries_without_json_mode(self):
        calls = []
        responses = [
            FakeResponse(200, ""),
            FakeResponse(200, '{"answers":["C"],"confidence":0.9}'),
        ]
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            client = self._client()
            with patch("requests.Session.post", side_effect=self._fake_post(calls, responses)):
                answer = client.answer(dummy_question())
        self.assertEqual(answer.letters, ["C"])
        self.assertEqual(len(calls), 2)
        self.assertIn("response_format", calls[0])
        self.assertNotIn("response_format", calls[1])


class VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        try:
            cls.ocr = OcrEngine()
            cls.ocr.warmup()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"OCR 不可用: {exc}")
        cls.scanner = PanelScanner(cls.ocr, cls.cfg)
        cls.player = PlayerMonitor(cls.ocr, cls.cfg)

    def test_ocr_thread_limits(self):
        self.assertEqual(self.ocr.intra_threads, 1)
        self.assertEqual(self.ocr.inter_threads, 1)
        self.assertEqual(self.ocr.cv_threads, 1)
        self.assertEqual(self.ocr.det_limit_side, 416)
        engine_cfg = self.ocr._engine.cfg
        self.assertEqual(int(engine_cfg.EngineConfig.onnxruntime.intra_op_num_threads), 1)
        self.assertEqual(int(engine_cfg.EngineConfig.onnxruntime.inter_op_num_threads), 1)
        self.assertEqual(int(engine_cfg.Det.limit_side_len), 416)
        self.assertFalse(bool(engine_cfg.Global.use_cls))

    def test_high_progress_bar(self):
        image = load("player_high_progress.png")
        layout = Layout.manual(
            image.shape[1], image.shape[0], panel_left=image.shape[1], panel_right=image.shape[1], panel_top=0
        )
        state = self.player.snapshot(image, layout)
        self.assertIsNotNone(state, "77% 进度的蓝色进度条未被识别")
        self.assertEqual(state.current, 500)
        self.assertEqual(state.total, 716)
        self.assertTrue(state.playing, f"应识别为播放中，实际 {state.playing}")

    def test_playing_4pct_with_taskbar(self):
        image = load("player_playing_4pct.png")
        layout = Layout.manual(
            image.shape[1], image.shape[0], panel_left=2046, panel_right=image.shape[1], panel_top=524
        )
        state = self.player.snapshot(image, layout)
        self.assertIsNotNone(state, "任务栏不应抢走真实进度条")
        self.assertEqual(state.current, 4)
        self.assertEqual(state.total, 713)
        self.assertTrue(state.playing)
        self.assertLess(state.bar_y, 1500, "选中的应是播放器进度条而不是任务栏")

    def test_next_button_is_not_treated_as_play(self):
        frame = np.zeros((200, 1200, 3), np.uint8)
        triangle = np.array([[100, 90], [100, 130], [130, 110]], np.int32)
        cv2.fillPoly(frame, [triangle], (255, 255, 255))
        cv2.rectangle(frame, (140, 90), (145, 130), (255, 255, 255), -1)
        playing, button = vision.detect_play_state(
            frame, 60, 1200, Rect(300, 95, 600, 125)
        )
        self.assertIsNone(button, "下一节按钮不能被当成播放按钮")
        self.assertIsNone(playing)

    def test_panel_ring_77_is_in_progress(self):
        image = load("panel_ring_77.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=0, panel_right=image.shape[1], panel_top=0)
        lessons = {l.index: l for l in self.scanner.scan(image, layout)}
        self.assertEqual(lessons["2.1"].status, "in_progress")
        self.assertEqual(lessons["2.1"].percent, 77)
        self.assertEqual(lessons["3.1"].status, "in_progress")
        self.assertEqual(lessons["3.1"].percent, 4)

    def test_paused_frame_bar_and_play_button(self):
        image = load("player_paused_836.jpg")
        layout = Layout.manual(
            image.shape[1], image.shape[0], panel_left=2046, panel_right=image.shape[1], panel_top=524
        )
        state = self.player.snapshot(image, layout)
        self.assertIsNotNone(state, "暂停画面也应能识别到进度条")
        self.assertEqual(state.current, 516)
        self.assertEqual(state.total, 716)
        self.assertIsNotNone(state.play_button)
        self.assertLess(state.play_button.width, 40, "播放按钮不应与下一节按钮合并")

    def test_panel_statuses(self):
        image = load("panel_chapter1.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=0, panel_right=image.shape[1], panel_top=0)
        lessons = self.scanner.scan(image, layout)
        statuses = {l.index: l.status for l in lessons}
        self.assertEqual(len(lessons), 5)
        self.assertEqual(statuses["1.1"], "completed")
        self.assertEqual(statuses["1.2"], "in_progress")
        self.assertEqual(statuses["1.3"], "not_started")
        self.assertTrue(all("测试" not in l.title for l in lessons))

    def test_full_page_ignores_ai_avatar(self):
        image = load("page_full.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=1017, panel_right=image.shape[1], panel_top=232)
        lessons = {l.index: l for l in self.scanner.scan(image, layout)}
        self.assertEqual(lessons["1.1"].status, "completed")
        self.assertEqual(lessons["1.2"].status, "in_progress")
        for index in ("1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4"):
            self.assertEqual(lessons[index].status, "not_started", index)

    def test_player_controls(self):
        image = load("player_controls.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=image.shape[1], panel_right=image.shape[1])
        state = self.player.snapshot(image, layout)
        self.assertIsNotNone(state)
        self.assertEqual(state.current, 15)
        self.assertEqual(state.total, 716)
        self.assertTrue(state.playing)
        self.assertIsNotNone(state.play_button)

    def test_bar_pixels(self):
        image = load("progress_bar.png")
        padded = np.vstack([np.full((520, image.shape[1], 3), 20, np.uint8), image])
        layout = Layout.manual(padded.shape[1], padded.shape[0], panel_left=padded.shape[1], panel_right=padded.shape[1])
        state = self.player.snapshot(padded, layout)
        self.assertIsNotNone(state)
        self.assertEqual(state.current, 43)
        self.assertEqual(state.total, 716)
        self.assertTrue(0.05 < state.ratio < 0.08)

    def test_no_bar_when_controls_hidden(self):
        image = load("page_full.png")
        self.assertIsNone(vision.find_progress_bar(image, 1017))


class HotkeyTests(unittest.TestCase):
    def test_parse_hotkey(self):
        self.assertEqual(parse_hotkey("F9"), 0x78)
        self.assertEqual(parse_hotkey("f1"), 0x70)
        self.assertEqual(parse_hotkey("esc"), 0x1B)
        self.assertIsNone(parse_hotkey("ctrl+q"))
        self.assertIsNone(parse_hotkey(""))

    def test_listener_start_stop_is_safe_and_sets_event(self):
        event = threading.Event()
        listener = HotkeyListener("F9", event, logging.getLogger("test.hotkey"))
        listener.start()
        listener.stop()
        listener._on_hotkey()
        self.assertTrue(event.is_set())


class QuizTests(unittest.TestCase):
    def test_close_button_ignores_unanswered_warning(self):
        dialog = Rect(0, 0, 500, 300)
        lines = [
            OcrLine(text="未作答的弹题不能关闭", rect=Rect(100, 200, 340, 220), score=0.95)
        ]
        self.assertIsNone(find_close_button(lines, dialog))

    def test_option_with_selected_bullet(self):
        dialog = Rect(0, 0, 500, 300)
        lines = [
            OcrLine(text="下列行列式的值为", rect=Rect(10, 10, 220, 30), score=0.95),
            OcrLine(text="● A. -42", rect=Rect(10, 50, 120, 70), score=0.95),
        ]
        q = parse_quiz(lines, dialog)
        self.assertEqual(q.options.get("A"), "-42")

    def test_quiz_wheel_is_split_into_small_events(self):
        cfg = load_config()
        mouse = FakeMouse()
        handler = QuizHandler(None, cfg, mouse, None, None, None)
        dialog = Rect(100, 100, 900, 700)
        handler._wheel_pixels(dialog, 300, -1)
        self.assertEqual(sum(click for _x, _y, click in mouse.scrolls), -300)
        self.assertGreaterEqual(len(mouse.scrolls), 5)
        x, _y = handler._body_point(dialog)
        self.assertGreater(x, dialog.x0 + 0.85 * dialog.width)

    def test_quiz_stitch_overlap(self):
        rng = np.random.default_rng(0)
        prev = rng.integers(0, 255, (160, 80, 3), dtype=np.uint8)
        new = np.vstack([prev[50:], np.full((60, 80, 3), 128, dtype=np.uint8)])
        out = QuizHandler._stitch_vertical(prev, new)
        self.assertEqual(out.shape[0], 220, "拼接应只追加重叠区以下的新内容")
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        try:
            cls.ocr = OcrEngine()
            cls.ocr.warmup()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"OCR 不可用: {exc}")

    def test_no_llm_video_quiz_random(self):
        cfg = load_config()
        cfg.llm.enabled = False
        notifier = FakeNotifier()
        frame = np.zeros((400, 900, 3), np.uint8)
        dialog = Rect(50, 50, 850, 350)
        q = QuizQuestion(
            kind="单选题",
            text="1+1=?",
            options={"A": "对", "B": "错"},
            option_rects={"A": Rect(100, 200, 120, 220), "B": Rect(100, 250, 120, 270)},
            dialog=dialog,
        )
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            state = State(Path(tmp) / "state.json")
            handler = QuizHandler(
                self.ocr, cfg, FakeMouse(FakeScreen(frame)), FakeScreen(frame), FailingLLM(),
                state, notifier=notifier,
            )
            answer = handler._ask_question(q, frame)
        self.assertEqual(answer.source, "fallback")
        self.assertIn(answer.letters[0], ["A", "B"])
        self.assertTrue(notifier.messages)
        self.assertFalse(cfg.llm_ready())

    def test_video_quiz_random_fallback_enabled(self):
        cfg = load_config()
        cfg.llm.fallback_random_enabled = True
        frame = np.zeros((400, 900, 3), np.uint8)
        dialog = Rect(50, 50, 850, 350)
        screen = FakeScreen(frame)
        mouse = FakeMouse(screen)
        notifier = FakeNotifier()
        q = QuizQuestion(
            kind="单选题",
            text="1+1=?",
            options={"A": "对", "B": "错"},
            option_rects={"A": Rect(100, 200, 120, 220), "B": Rect(100, 250, 120, 270)},
            dialog=dialog,
        )
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            state = State(Path(tmp) / "state.json")
            handler = QuizHandler(
                self.ocr, cfg, mouse, screen, FailingLLM(), state, notifier=notifier
            )
            handler.is_answered = lambda frame, dialog: (False, [])
            handler.collect = lambda dialog: q
            handler.close = lambda frame, dialog: True
            applied = []
            handler.apply_answer = lambda dialog, letters, single: applied.extend(letters) or True
            result = handler.solve(frame, dialog)
        self.assertTrue(result.get("fallback"))
        self.assertTrue(notifier.messages, "视频弹题随机兜底必须通知人工复核")
        self.assertTrue(applied)
        self.assertIn(applied[0], ["A", "B"])

    def _question(self, name: str, exclude_x: int, upscale: float = 1.4):
        image = load(name)
        cards = vision.find_cards(
            image, exclude_x, self.cfg.overlay.card_threshold, self.cfg.overlay.card_close_kernel
        )
        self.assertTrue(cards, "未找到弹窗")
        card = cards[0]
        lines = self.ocr.read(card.crop(image), offset=(card.x0, card.y0), upscale=upscale)
        return image, card, lines, parse_quiz(lines, card, image)

    def test_multi_select(self):
        _image, _card, _lines, question = self._question("quiz_multi.png", 10000)
        self.assertEqual(question.kind, "多选题")
        self.assertEqual(set(question.options), {"A", "B", "C", "D"})
        self.assertIn("资政新篇", question.text)
        self.assertFalse(question.needs_vision)

    def test_math_needs_vision(self):
        _image, _card, _lines, question = self._question("quiz_math.png", 1017, 2.0)
        self.assertTrue(question.needs_vision)
        self.assertIn("A", question.options)

    def test_answered(self):
        image = load("quiz_answered.png")
        cards = vision.find_cards(image, 10000, self.cfg.overlay.card_threshold, self.cfg.overlay.card_close_kernel)
        card = cards[0]
        lines = self.ocr.read(card.crop(image), offset=(card.x0, card.y0), upscale=1.4)
        letters, correct = parse_feedback("\n".join(l.text for l in lines))
        self.assertEqual(letters, ["C"])
        self.assertTrue(correct)

    def test_three_option_question_is_complete_without_phantom_d(self):
        from types import SimpleNamespace

        q = SimpleNamespace(kind="单选题", options={"A": "1", "B": "2", "C": "3"})
        self.assertEqual(QuizHandler._expected_letters(q), ["A", "B", "C"])

    def test_multi_image_content_has_one_url_per_screenshot(self):
        content = LLMClient._user_content("题目", ["aaa", "bbb", "ccc"])
        self.assertEqual(content[0]["type"], "text")
        images = [x for x in content if x.get("type") == "image_url"]
        self.assertEqual(len(images), 3)
        self.assertTrue(all("data:image/jpeg;base64," in x["image_url"]["url"] for x in images))

    def test_first_screen_with_missing_c_is_not_complete(self):
        from types import SimpleNamespace

        dialog = Rect(0, 0, 1000, 800)
        close = OcrLine("关闭", Rect(450, 700, 550, 740))
        q = SimpleNamespace(kind="单选题", options={"A": "-8", "B": "8", "D": "32"})
        self.assertFalse(QuizHandler._complete(q, [close], dialog))
        q.options["C"] = "-32"
        self.assertTrue(QuizHandler._complete(q, [close], dialog))

    def test_option_selection_ignores_blue_formula(self):
        frame = np.full((400, 800, 3), 255, np.uint8)
        # 选项文字/公式区域有蓝色，但左侧圆点仍是灰色
        cv2.rectangle(frame, (300, 160), (500, 230), (200, 80, 30), -1)
        cv2.circle(frame, (210, 200), 14, (210, 210, 210), -1)
        handler = QuizHandler(self.ocr, self.cfg, None, None, None, None)
        rect = Rect(300, 170, 500, 230)
        self.assertFalse(handler._selected(frame, rect, checkbox_x=210))
        # 圆点变成主题蓝后才算选中
        cv2.circle(frame, (210, 200), 14, (220, 120, 40), -1)
        self.assertTrue(handler._selected(frame, rect, checkbox_x=210))

    def test_unanswered_prompt_closes_its_own_x(self):
        image = load("unanswered_prompt.png")
        dialog = Rect(120, 75, 1070, 795)

        class Screen:
            def grab(self):
                return image

        class Mouse:
            def __init__(self):
                self.clicks = []

            def click(self, x, y, label=""):
                self.clicks.append((x, y, label))

            def click_rect(self, rect, label=""):
                self.clicks.append((rect.cx, rect.cy, label))

            def press(self, key):
                self.clicks.append((0, 0, f"key:{key}"))

        mouse = Mouse()
        handler = QuizHandler(self.ocr, self.cfg, mouse, Screen(), None, None)
        prompt = handler._unanswered_prompt(dialog)
        self.assertIsNotNone(prompt)
        self.assertLess(prompt.width, dialog.width)
        handler._dismiss_unanswered(dialog)
        self.assertTrue(mouse.clicks)
        x, _y, _label = mouse.clicks[0]
        # 正确目标是提示卡右上角的 X，而不是外层弹题右上角的 X
        self.assertLess(x, dialog.cx + int(0.25 * dialog.width))
        self.assertGreater(x, dialog.cx)

    def test_option_selection_detection(self):
        image = load("quiz_answered.png")
        cards = vision.find_cards(image, 10000, self.cfg.overlay.card_threshold, self.cfg.overlay.card_close_kernel)
        card = cards[0]
        lines = self.ocr.read(card.crop(image), offset=(card.x0, card.y0), upscale=1.4)
        question = parse_quiz(lines, card, image)
        handler = QuizHandler(self.ocr, self.cfg, None, None, None, None)
        checkbox_x = card.x0 + int(0.17 * card.width)
        self.assertTrue(handler._selected(image, question.option_rects["C"], checkbox_x))
        self.assertFalse(handler._selected(image, question.option_rects["A"], checkbox_x))
        self.assertIsNotNone(find_close_button(lines, card))

    def test_must_read_and_daily(self):
        for name, expected in (("popup_mustread.png", "must_read"), ("popup_daily.png", "daily")):
            image = load(name)
            cards = vision.find_cards(
                image, 10000, self.cfg.overlay.card_threshold, self.cfg.overlay.card_close_kernel
            )
            self.assertTrue(cards, name)
            card = cards[0]
            header = self.ocr.read_text(
                Rect(card.x0, card.y0, card.x1, card.y0 + max(90, int(0.28 * card.height))).crop(image),
                offset=(card.x0, card.y0),
                upscale=1.3,
            )
            self.assertEqual(AutoLesson._classify_overlay(header), expected, header)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class FakeScreen:
    def __init__(self, frame):
        self.frame = frame
        self.after = None

    def grab(self):
        return self.frame

    def grab_rect(self, rect):
        return self.frame[rect.y0 : rect.y1, rect.x0 : rect.x1]

    def save(self, path, frame):
        return True

    def close(self):
        pass


class FakeMouse:
    def __init__(self, screen=None):
        self.screen = screen
        self.clicks = []
        self.scrolls = []
        self.moves = []
        self.moves = []

    def move_to(self, x, y, duration=0.25):
        self.moves.append((x, y))

    def click(self, x, y, label=""):
        self.clicks.append((x, y, label))
        if self.screen is not None and self.screen.after is not None:
            self.screen.frame, self.screen.after = self.screen.after, None

    def click_rect(self, rect, label=""):
        self.click(rect.cx, rect.cy, label)

    def press(self, key):
        pass

    def scroll_at(self, x, y, clicks):
        self.scrolls.append((x, y, clicks))

    def wiggle(self, x, y):
        pass

    def move_to(self, x, y, duration=0.25):
        self.moves.append((x, y))

    def drag(self, x0, y0, x1, y1, label=""):
        self.clicks.append((x1, y1, label))

    def hotkey(self, *keys):
        self.clicks.append(("hotkey",) + tuple(keys))


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send(self, *args, **kwargs):
        self.messages.append((args, kwargs))


class FakeQuiz:
    def __init__(self):
        self.calls = 0

    def solve(self, frame, dialog):
        self.calls += 1
        return {"answered": True, "closed": True}


class ScrollingTests(unittest.TestCase):
    def test_scroll_ignores_right_side_animation(self):
        cfg = load_config()
        frame = np.zeros((400, 1000, 3), np.uint8)
        frame[100:380, 800:1000] = 255

        class AnimatedIconScreen(FakeScreen):
            def __init__(self, image):
                super().__init__(image)
                self.n = 0

            def grab_rect(self, rect):
                image = super().grab_rect(rect).copy()
                self.n += 1
                if image.shape[1] > 70:
                    image[:, -62:] = (self.n * 37) % 255
                return image

        screen = AnimatedIconScreen(frame)
        mouse = FakeMouse()
        layout = Layout.manual(1000, 400, panel_left=800, panel_right=1000, panel_top=50)
        navigator = PanelNavigator(None, None, cfg, mouse, screen, layout)
        # 右侧状态图标/滚动条在动，但左侧目录文字没动 -> 不应判定为滚动
        self.assertFalse(navigator.scroll_down())

    def test_scroll_uses_signature_without_ocr(self):
        cfg = load_config()
        frame = np.zeros((400, 1000, 3), np.uint8)
        frame[100:380, 800:1000] = 255
        screen = FakeScreen(frame)
        mouse = FakeMouse()
        layout = Layout.manual(1000, 400, panel_left=800, panel_right=1000, panel_top=50)
        # scanner/ocr 传 None：一旦滚动逻辑仍调用 OCR，这里会直接报错
        navigator = PanelNavigator(None, None, cfg, mouse, screen, layout)
        navigator.scroll_to_top()
        self.assertGreaterEqual(len(mouse.scrolls), 2)
        self.assertFalse(navigator.scroll_down())


class PageScanTests(unittest.TestCase):
    def test_scan_all_reaches_bottom_and_returns_top(self):
        cfg = load_config()
        rng = np.random.default_rng(7)
        content = np.full((2000, 300, 3), 245, np.uint8)
        content += rng.integers(0, 20, content.shape, dtype=np.uint8)
        panel = Rect(0, 0, 300, 400)
        layout = Layout.manual(300, 400, panel_left=0, panel_right=300, panel_top=0)

        class ScrollScreen:
            def __init__(self):
                self.offset = 0

            @property
            def max_offset(self):
                return content.shape[0] - panel.height

            def grab_rect(self, rect):
                return content[self.offset : self.offset + rect.height, rect.x0 : rect.x1].copy()

            def grab(self):
                return self.grab_rect(panel)

        class ScrollMouse:
            def __init__(self, screen):
                self.screen = screen
                self.scrolls = []
                self.max_seen = 0

            def scroll_at(self, x, y, amount):
                self.scrolls.append(amount)
                self.screen.offset = min(
                    self.screen.max_offset,
                    max(0, self.screen.offset - int(amount)),
                )
                self.max_seen = max(self.max_seen, self.screen.offset)

        class OffsetScanner:
            def __init__(self, screen):
                self.screen = screen

            def scan(self, frame, layout):
                return [
                    Lesson(
                        index=f"1.{self.screen.offset}",
                        title=f"row {self.screen.offset}",
                        duration=60,
                        status="not_started",
                        title_rect=Rect(0, 10, 200, 30),
                        row=Rect(0, 0, 300, 60),
                    )
                ]

        screen = ScrollScreen()
        mouse = ScrollMouse(screen)
        navigator = PanelNavigator(OffsetScanner(screen), None, cfg, mouse, screen, layout)
        lessons = navigator.scan_all()
        self.assertTrue(navigator.last_scan_complete)
        self.assertEqual(mouse.max_seen, screen.max_offset)
        self.assertGreater(len(lessons), 3)
        self.assertEqual(screen.offset, 0, "扫描结束应回到顶部")


class ControllerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.cfg.quiz.enabled = False
        cls.cfg.notify.desktop = False
        cls.cfg.notify.sound = False
        try:
            cls.ocr = OcrEngine()
            cls.ocr.warmup()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"OCR 不可用: {exc}")

    def _app(self, frame, layout, tempdir):
        screen = FakeScreen(frame)
        mouse = FakeMouse(screen)
        state = State(Path(tempdir) / "state.json")
        app = AutoLesson(self.cfg, self.ocr, screen, mouse, state, FakeNotifier())
        app.layout = layout
        return app, screen, mouse

    def test_normal_page_has_no_overlay(self):
        image = load("page_full.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=1017, panel_right=image.shape[1], panel_top=232)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, mouse = self._app(image, layout, tmp)
            self.assertFalse(app._handle_overlays(image))
            self.assertEqual(mouse.clicks, [])

    def test_ai_floating_icon_absent_on_blank_page(self):
        from types import SimpleNamespace
        from autolesson.layout import Layout

        frame = np.full((1600, 2560, 3), 255, np.uint8)
        layout = Layout.manual(2560, 1600, panel_left=2046, panel_right=2560, panel_top=525)
        fake = SimpleNamespace(
            layout=layout,
            log=logging.getLogger("test.ai.absent"),
            _ai_icon_template=AutoLesson._ai_icon_template,
        )
        self.assertIsNone(AutoLesson._find_ai_icon(fake, frame))

    def test_ai_floating_ball_dragged_directly(self):
        from types import MethodType, SimpleNamespace
        from autolesson.layout import Layout

        icon = cv2.imread(str(ROOT / "autolesson" / "assets" / "ai_assistant_icon.png"), cv2.IMREAD_UNCHANGED)
        icon = cv2.resize(icon, (160, 160), interpolation=cv2.INTER_AREA)
        frame = np.full((1600, 2560, 3), 255, np.uint8)
        alpha = icon[:, :, 3:4].astype(np.float32) / 255.0
        frame[900:1060, 2200:2360] = (
            icon[:, :, :3].astype(np.float32) * alpha + 255 * (1 - alpha)
        ).astype(np.uint8)

        class Mouse:
            def __init__(self):
                self.drags = []

            def drag(self, x0, y0, x1, y1, label=""):
                self.drags.append((x0, y0, x1, y1, label))

            def click_rect(self, *args, **kwargs):
                raise AssertionError("不应点击 X，应该直接拖动")

        layout = Layout.manual(2560, 1600, panel_left=2046, panel_right=2560, panel_top=525)
        fake = SimpleNamespace(
            cfg=load_config(),
            layout=layout,
            log=logging.getLogger("test.ai.drag"),
            ocr=SimpleNamespace(read=lambda frame: []),
            mouse=Mouse(),
            _sleep=lambda seconds: None,
            _grab=lambda label="": frame,
            _find_x_in=AutoLesson._find_x_in,
            _ai_icon_template=AutoLesson._ai_icon_template,
        )
        fake._find_ai_icon = MethodType(AutoLesson._find_ai_icon, fake)
        AutoLesson._prepare_ai_assistant(fake)
        self.assertTrue(fake.mouse.drags, "检测到悬浮球时应直接拖动")
        _x0, _y0, _x1, target_y, _label = fake.mouse.drags[0]
        self.assertLess(target_y, layout.panel_top)
        self.assertGreater(target_y, 100)

    def test_ai_floating_icon_template_match(self):
        from types import SimpleNamespace
        from autolesson.layout import Layout

        icon = cv2.imread(str(ROOT / "autolesson" / "assets" / "ai_assistant_icon.png"), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(icon)
        icon = cv2.resize(icon, (160, 160), interpolation=cv2.INTER_AREA)
        frame = np.full((1600, 2560, 3), 255, np.uint8)
        alpha = icon[:, :, 3:4].astype(np.float32) / 255.0
        frame[800:960, 2200:2360] = (
            icon[:, :, :3].astype(np.float32) * alpha + 255 * (1 - alpha)
        ).astype(np.uint8)
        layout = Layout.manual(2560, 1600, panel_left=2046, panel_right=2560, panel_top=525)
        fake = SimpleNamespace(
            layout=layout,
            log=logging.getLogger("test.ai"),
            _ai_icon_template=AutoLesson._ai_icon_template,
        )
        rect = AutoLesson._find_ai_icon(fake, frame)
        self.assertIsNotNone(rect)
        self.assertLess(abs(rect.cx - 2280), 20)
        self.assertLess(abs(rect.cy - 880), 20)

    def test_ai_assistant_does_not_depend_on_student_name(self):
        lines = [OcrLine("示例同学", Rect(10, 10, 120, 30))]
        self.assertEqual(AutoLesson._ai_text(lines), "")
        lines.append(OcrLine("每日6点更新", Rect(10, 40, 120, 60)))
        self.assertIn("每日6点更新", AutoLesson._ai_text(lines))

    def test_known_completed_skips_transient_not_started(self):
        image = load("page_full.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=1017, panel_right=image.shape[1], panel_top=232)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, _mouse = self._app(image, layout, tmp)
            completed = Lesson(
                index="0.1",
                title="课程的性质定位及结构主线",
                duration=638,
                status="completed",
                title_rect=Rect(2280, 610, 2460, 630),
                row=Rect(2046, 590, 2560, 650),
            )
            app._remember_completed(completed)
            stale = Lesson(
                index="0.1",
                title="课程的性质定位及结构主线",
                duration=638,
                status="not_started",
                title_rect=Rect(2280, 610, 2460, 630),
                row=Rect(2046, 590, 2560, 650),
            )
            self.assertFalse(app._is_pending(stale))
            stale.status = "in_progress"
            stale.percent = 2
            self.assertFalse(app._is_pending(stale))

    def test_retry_retries_same_lesson(self):
        image = load("page_full.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=1017, panel_right=image.shape[1], panel_top=232)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, _mouse = self._app(image, layout, tmp)
            lesson = Lesson(
                index="2.1",
                title="农民阶级的天国梦及其局限",
                duration=716,
                status="in_progress",
                title_rect=Rect(2280, 1190, 2400, 1210),
                row=Rect(2046, 1170, 2560, 1230),
            )
            calls = []

            def fake_process(current):
                calls.append(current.key)
                if len(calls) == 1:
                    raise LessonError("player not loaded")

            app._process_lesson = fake_process
            app._find_lesson_anywhere = lambda current: current
            app._sleep = lambda seconds: None
            result = app._process_with_retry(lesson)
            self.assertEqual(calls, [lesson.key, lesson.key])
            self.assertEqual(result.key, lesson.key)

    def test_lesson_order_by_section_index(self):
        from types import SimpleNamespace

        a = SimpleNamespace(index="5.8", title="循环结构举例(2)")
        b = SimpleNamespace(index="2.2", title="程序输入输出")
        self.assertLess(AutoLesson._lesson_order(b), AutoLesson._lesson_order(a))

    def test_verify_loaded_reveals_controls_without_click(self):
        frame = np.zeros((400, 900, 3), np.uint8)
        layout = Layout.manual(900, 400, panel_left=700, panel_right=900, panel_top=100)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, mouse = self._app(frame, layout, tmp)
            state = PlaybackState(
                current=516, total=716, ratio=516 / 716, playing=False, bar_y=1250, play_button=None
            )

            class FakePlayer:
                def __init__(self):
                    self.ready = False
                    self.shown = 0

                def refresh(self, *args, **kwargs):
                    return state if self.ready else None

                def show_controls(self, mouse, layout):
                    self.shown += 1
                    self.ready = True

            fake = FakePlayer()
            app.player = fake
            app.mouse = mouse
            lesson = Lesson(
                index="2.1",
                title="农民阶级的天国梦及其局限",
                duration=716,
                status="in_progress",
                title_rect=Rect(2280, 1190, 2400, 1210),
                row=Rect(2046, 1170, 2560, 1230),
            )
            app._verify_loaded(lesson)
            self.assertTrue(fake.ready)
            self.assertGreater(fake.shown, 0)
            self.assertFalse(mouse.clicks, "控件隐藏时应移动鼠标唤出，不能点击视频画面")

    def test_pending_skips_recorded_lesson(self):
        frame = np.zeros((400, 900, 3), np.uint8)
        layout = Layout.manual(900, 400, panel_left=700, panel_right=900, panel_top=100)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, _mouse = self._app(frame, layout, tmp)
            lesson = Lesson(
                index="2.2",
                title="程序输入输出",
                duration=1133,
                status="not_started",
                title_rect=Rect(0, 0, 100, 20),
                row=Rect(0, 0, 150, 40),
            )
            app.state.mark_completed(lesson)
            self.assertFalse(app._is_pending(lesson))

    def test_ensure_playing_clicks_video_center_first(self):
        frame = np.zeros((400, 900, 3), np.uint8)
        layout = Layout.manual(900, 400, panel_left=700, panel_right=900, panel_top=100)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, mouse = self._app(frame, layout, tmp)
            paused = PlaybackState(
                current=139, total=629, ratio=139 / 629, playing=False, bar_y=250, play_button=None
            )
            advanced = PlaybackState(
                current=142, total=629, ratio=142 / 629, playing=True, bar_y=250, play_button=None
            )

            class FakePlayer:
                def __init__(self):
                    self.waits = 0
                    self.clicks = 0

                def refresh(self, *args, **kwargs):
                    return paused

                def show_controls(self, *args, **kwargs):
                    pass

                def wait_for_advance(self, *args, **kwargs):
                    self.waits += 1
                    return None if self.waits == 1 else advanced

                def click_play(self, *args, **kwargs):
                    self.clicks += 1

            fake = FakePlayer()
            app.player = fake
            app.mouse = mouse
            result = app._ensure_playing()
            self.assertEqual(result.current, 142)
            self.assertTrue(mouse.clicks)
            self.assertEqual(mouse.clicks[0][1], int(layout.screen_h * 0.50))
            self.assertEqual(fake.clicks, 0, "中央点击成功后不应再点播放键")

    def test_ensure_playing_does_not_click_when_progressing(self):
        frame = np.zeros((400, 900, 3), np.uint8)
        layout = Layout.manual(900, 400, panel_left=700, panel_right=900, panel_top=100)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, _mouse = self._app(frame, layout, tmp)
            button = Rect(5, 287, 25, 313)
            paused_icon = PlaybackState(
                current=516, total=716, ratio=516 / 716, playing=False, bar_y=250, play_button=button
            )
            advanced_state = PlaybackState(
                current=520, total=716, ratio=520 / 716, playing=True, bar_y=250, play_button=button
            )

            class FakePlayer:
                def refresh(self, *args, **kwargs):
                    return paused_icon

                def wait_for_advance(self, *args, **kwargs):
                    return advanced_state

                def click_play(self, *args, **kwargs):
                    raise AssertionError("进度已经在前进，不应该再点播放")

                def video_center(self, layout):
                    return (100, 100)

            app.player = FakePlayer()
            result = app._ensure_playing()
            # 图标误判为暂停，但时间在前进：绝不能点播放，否则会把视频按停
            self.assertEqual(result.current, 520)

    def test_resume_after_overlay_clicks_once_when_paused(self):
        frame = np.zeros((400, 900, 3), np.uint8)
        layout = Layout.manual(900, 400, panel_left=700, panel_right=900, panel_top=100)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, mouse = self._app(frame, layout, tmp)
            button = Rect(5, 200, 25, 230)
            state = PlaybackState(
                current=516, total=716, ratio=516 / 716, playing=True, bar_y=250, play_button=button
            )

            class FakePlayer:
                def __init__(self):
                    self.clicks = 0

                def refresh(self, *args, **kwargs):
                    return state

                def wait_for_advance(self, *args, **kwargs):
                    return state

                def click_play(self, *args, **kwargs):
                    self.clicks += 1

                def video_center(self, layout):
                    return (100, 100)

            fake = FakePlayer()
            app.player = fake
            app._sleep = lambda seconds: None
            app._resume_after_overlay()
            # 答题后视频仍暂停：只点一次播放键
            self.assertEqual(fake.clicks, 1)
            self.assertTrue(mouse.moves)

    def test_click_play_once_only_clicks_play_button(self):
        frame = np.zeros((400, 900, 3), np.uint8)
        layout = Layout.manual(900, 400, panel_left=700, panel_right=900, panel_top=100)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, mouse = self._app(frame, layout, tmp)
            button = Rect(5, 200, 25, 230)
            state = PlaybackState(
                current=516, total=716, ratio=516 / 716, playing=False, bar_y=250, play_button=button
            )
            clicked = app._click_play_once(state)
            self.assertTrue(clicked)
            self.assertEqual(len(mouse.clicks), 1)
            self.assertEqual(mouse.clicks[0][:2], (button.cx, button.cy))
            self.assertTrue(mouse.moves, "点播放前应先把鼠标移到视频中央")

    def test_ensure_playing_clicks_upper_then_advances(self):
        frame = np.zeros((400, 900, 3), np.uint8)
        layout = Layout.manual(900, 400, panel_left=700, panel_right=900, panel_top=100)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, mouse = self._app(frame, layout, tmp)
            button = Rect(5, 1287, 25, 1313)
            first = PlaybackState(
                current=516, total=716, ratio=516 / 716, playing=True, bar_y=1250, play_button=button
            )

            class FakePlayer:
                def __init__(self):
                    self.clicks = 0
                    self.waits = 0

                def refresh(self, *args, **kwargs):
                    return first

                def wait_for_advance(self, *args, **kwargs):
                    self.waits += 1
                    if self.waits == 1:
                        return first
                    return PlaybackState(
                        current=520, total=716, ratio=520 / 716, playing=True, bar_y=1250, play_button=button
                    )

                def click_play(self, *args, **kwargs):
                    self.clicks += 1

            fake = FakePlayer()
            app.player = fake
            result = app._ensure_playing()
            self.assertEqual(result.current, 520)
            self.assertEqual(fake.clicks, 0, "点击视频中央成功后不应再点播放键")
            self.assertTrue(mouse.clicks)
            self.assertEqual(mouse.clicks[0][1], int(layout.screen_h * 0.50))

    def test_quiz_dispatches_to_handler(self):
        image = load("quiz_full.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=2050, panel_right=image.shape[1], panel_top=454)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, _screen, _mouse = self._app(image, layout, tmp)
            fake = FakeQuiz()
            app.quiz = fake
            self.assertTrue(app._handle_overlays(image))
            self.assertEqual(fake.calls, 1)

    def test_must_read_can_be_closed(self):
        image = load("popup_mustread.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=image.shape[1], panel_right=image.shape[1], panel_top=0)
        after = np.zeros_like(image)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            app, screen, mouse = self._app(image, layout, tmp)
            screen.after = after
            self.assertTrue(app._handle_overlays(image))
            self.assertTrue(mouse.clicks, "应至少点击一次关闭按钮")


class PracticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.cfg.practice.question_wait = 0.01
        try:
            cls.ocr = OcrEngine()
            cls.ocr.warmup()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"OCR 不可用: {exc}")

    def _page_image(self, image, llm=None, notifier=None):
        screen = FakeScreen(image)
        mouse = FakeMouse(screen)
        tmp = tempfile.TemporaryDirectory(dir=RUNTIME)
        self.addCleanup(tmp.cleanup)
        state = State(Path(tmp.name) / "state.json")
        page = PracticePage(
            self.ocr, self.cfg, mouse, screen, llm or FakeLLM(), state, notifier=notifier
        )
        return page, image, mouse

    def _page(self, name, llm=None):
        return self._page_image(load(name), llm)

    def test_question_count_meta_and_last_save_button(self):
        page, image, _mouse = self._page("practice_test_3q_last.png")
        lines = page._read(image)
        text = page._text(lines)
        self.assertEqual(page._question_count(lines, text, image), 3)
        self.assertIsNotNone(page._find_save(lines))
        # 顶部暂存必须存在；底部“保存”不能当顶层暂存
        saves = [ln.text.strip() for ln in lines if "暂存" in ln.text]
        self.assertTrue(any("暂存作业" in x for x in saves), saves)

    def test_list_mode_merges_video_and_test_order(self):
        from types import SimpleNamespace

        def video(y, key):
            return SimpleNamespace(title_rect=Rect(0, y, 200, y + 20), key=key)

        def test_entry(y, key):
            return PracticeEntry(
                chapter="第一章", title="平时测试", rect=Rect(0, y, 200, y + 20), key=key
            )

        lessons = [video(300, "v3"), video(100, "v1"), video(200, "v2")]
        tests = [test_entry(250, "t1"), test_entry(50, "t0")]
        items = merge_pending_items(lessons, tests, lambda lesson: True, set())
        self.assertEqual([x[3].key for x in items], ["t0", "v1", "v2", "t1", "v3"])

    def test_auto_command_parser(self):
        from autolesson.cli import build_parser

        args = build_parser().parse_args(["auto"])
        self.assertTrue(args.list_mode)
        self.assertEqual(args.command, "auto")

    def test_player_time_fallback_on_blue_slide(self):
        from autolesson.player import PlayerMonitor

        image = load("player_blue_slide_paused.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=2046, panel_top=524)
        state = PlayerMonitor(self.ocr, load_config()).snapshot(image, layout)
        self.assertIsNotNone(state, "蓝色幻灯片遮住进度条时应使用时间文字兜底")
        self.assertAlmostEqual(state.current, 139, delta=2)
        self.assertAlmostEqual(state.total, 629, delta=2)
        self.assertFalse(state.playing)
        self.assertGreater(state.ratio, 0.19)

    def test_state_completion_records_and_clear(self):
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            state = State(Path(tmp) / "state.json")
            lesson = Lesson(
                index="2.2",
                title="程序输入输出",
                duration=1133,
                status="not_started",
                title_rect=Rect(0, 0, 200, 20),
                row=Rect(0, 0, 300, 40),
            )
            state.mark_completed(lesson)
            state.mark_practice("第二章", "第二章", "平时测试")
            self.assertTrue(state.is_completed(lesson))
            self.assertTrue(state.is_practice_done("第二章"))
            state.clear_records()
            self.assertFalse(state.is_completed(lesson))
            self.assertFalse(state.is_practice_done("第二章"))

    def test_llm_empty_content_retries_then_answers(self):
        cfg = load_config()
        cfg.llm.api_key = "sk-test"
        cfg.llm.cache = False
        cfg.llm.empty_retries = 1
        cfg.llm.empty_retry_wait = 0.0
        q = QuizQuestion(
            kind="单选题",
            text="1+1=?",
            options={"A": "1", "B": "2"},
            option_rects={},
            dialog=Rect(0, 0, 100, 100),
        )
        responses = [
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": '{"answers":["B"],"confidence":0.9,"reason":"test"}'}}]},
        ]
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            state = State(Path(tmp) / "state.json")
            client = LLMClient(cfg, state)
            client._post = lambda *args, **kwargs: responses.pop(0)
            answer = client.answer(q)
        self.assertEqual(answer.letters, ["B"])

    def test_practice_no_random_fallback_on_llm_failure(self):
        cfg = self.cfg
        old = cfg.llm.fallback_random_enabled
        cfg.llm.fallback_random_enabled = True
        try:
            notifier = FakeNotifier()
            page, _image, mouse = self._page_image(
                load("exam_practice_replica.png"), llm=FailingLLM(), notifier=notifier
            )
            result = page.answer_current()
        finally:
            cfg.llm.fallback_random_enabled = old
        self.assertEqual(result, "skipped")
        self.assertTrue(notifier.messages, "跳过题目必须通知人工复核")
        self.assertIn("跳过", str(notifier.messages[-1]))
        self.assertFalse(
            any(c and c[0] == "click" for c in mouse.clicks),
            "练习不启用随机兜底，失败时不能选择任何选项",
        )

    def test_player_controls_revealed_without_click(self):
        from autolesson.player import PlayerMonitor

        screen = FakeScreen(np.zeros((1600, 2560, 3), dtype=np.uint8))
        mouse = FakeMouse(screen)
        layout = Layout.manual(2560, 1600, panel_left=2046, panel_top=604)
        mon = PlayerMonitor(self.ocr, load_config())
        mon.show_controls(mouse, layout)
        self.assertTrue(mouse.moves, "应移动鼠标唤出控件")
        self.assertLess(abs(mouse.moves[0][1] - 800), 120, "鼠标应停在视频中央附近")
        self.assertFalse(any(c and c[0] == "click" for c in mouse.clicks))

    def test_image_viewer_detected_and_closed(self):
        page, image, mouse = self._page("practice_image_viewer.png")
        self.assertTrue(page._image_viewer_open(image))
        self.assertTrue(page.close_image_viewer())
        self.assertTrue(
            any(c and c[0] == "hotkey" for c in mouse.clicks),
            "图片查看器应优先按 Esc 退出",
        )

    def test_practice_read_question_merges_options_after_scroll(self):
        from types import SimpleNamespace
        from autolesson.practice import PracticePage
        from autolesson.types import Rect

        first = SimpleNamespace(
            text="长题干",
            kind="单选题",
            options={"A": "甲", "B": "乙"},
            option_rects={"A": Rect(100, 100, 200, 120), "B": Rect(100, 200, 200, 220)},
        )
        second = SimpleNamespace(
            text="长题干",
            kind="单选题",
            options={"C": "丙", "D": "丁"},
            option_rects={"C": Rect(100, 700, 200, 720), "D": Rect(100, 800, 200, 820)},
        )
        page = object.__new__(PracticePage)
        page.screen = SimpleNamespace(grab=lambda: np.zeros((1000, 1200, 3), np.uint8))
        calls = []

        def read_once(frame):
            calls.append(1)
            return first if len(calls) == 1 else second

        page._read_question_once = read_once
        page._page_scroll = lambda direction=-1, pixels=360: len(calls) < 2
        q = PracticePage.read_question(page)
        self.assertEqual(sorted(q.options), ["A", "B", "C", "D"])
        self.assertIn("D", q.option_rects)

    def test_feedback_badge_with_type_suffix(self):
        from autolesson.quiz import is_feedback_text

        self.assertTrue(is_feedback_text("正确 【单选题】"))
        self.assertTrue(is_feedback_text("错误 【单选题】"))
        self.assertFalse(is_feedback_text("以下哪项回答正确？\nA. 甲"))

    def test_practice_saves_even_when_next_question_fails(self):
        llm = FakeLLM(["D"])
        page, _image, mouse = self._page("exam_practice_replica.png", llm)
        page.next_question = lambda count, index: False
        result = page.run(close_tab=False)
        self.assertTrue(result.get("saved"), "少做一题也必须保留已答内容并暂存")
        labels = [str(c[2]) for c in mouse.clicks if len(c) >= 3]
        self.assertIn("暂存作业", labels)
        self.assertFalse(any("提交" in x for x in labels), labels)

    def test_answer_card_two_rows_missing_number_keeps_row(self):
        from autolesson.practice import PracticePage
        from autolesson.types import OcrLine, Rect

        lines = []
        for n in (1, 3, 4, 5):
            x = 2100 + (n - 1) * 90
            lines.append(OcrLine(str(n), Rect(x, 800, x + 24, 824)))
        for n in (6, 7, 8, 9, 10):
            x = 2100 + (n - 6) * 90
            lines.append(OcrLine(str(n), Rect(x, 880, x + 24, 904)))

        class FakeOcr:
            def read(self, crop, offset=(0, 0), upscale=1.0):
                return lines

        page = object.__new__(PracticePage)
        page.ocr = FakeOcr()
        frame = np.zeros((1200, 2560, 3), np.uint8)
        numbers = page._answer_card_numbers(frame, 10)
        self.assertIn(2, numbers)
        self.assertLess(numbers[2].cy, 850, "第2题应补在第一行，不能按竖排算到下面")
        self.assertAlmostEqual(numbers[2].cx, 2190, delta=12)

    def test_practice_current_question_number_from_ocr(self):
        page, image, _mouse = self._page("practice_three.png")
        self.assertEqual(page._current_question_number(image), 2)

    def test_practice_three_option_question(self):
        page, image, _mouse = self._page("practice_three.png")
        q = page.read_question(image)
        self.assertIsNotNone(q)
        self.assertEqual(sorted(q.options), ["A", "B", "C"])
        self.assertNotIn("D", q.options, "三选项题不能凭空补 D")

    def test_option_click_uses_radio_not_image(self):
        page, image, mouse = self._page("exam_practice_replica.png")
        q = page.read_question(image)
        self.assertIsNotNone(q)
        page.apply_answer(image, q, ["A"])
        clicks = [c for c in mouse.clicks if len(c) >= 3]
        self.assertTrue(clicks)
        self.assertLess(
            clicks[0][0], q.option_rects["A"].x0,
            "选项点击应落在左侧圆点，不能点选项文字中心/题目图片",
        )

    def test_completed_test_row_skipped(self):
        image = load("practice_test_panel_completed.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=0, panel_top=0)
        entries = PracticeScanner(self.ocr, self.cfg).scan(image, layout)
        tests = [e for e in entries if "平时测试" in e.title]
        self.assertTrue(tests, "应识别到平时测试行")
        self.assertTrue(tests[0].completed, "平时测试行右侧蓝勾应识别为已完成")
        self.assertEqual(tests[0].chapter, "第一章")

    def test_runner_skips_completed_before_click(self):
        image = load("practice_test_panel_completed.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=0, panel_top=0)
        screen = FakeScreen(image)
        mouse = FakeMouse(screen)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as tmp:
            state = State(Path(tmp) / "state.json")

            class _Nav:
                def __init__(self, layout):
                    self.layout = layout

                def scroll_to_top(self):
                    pass

                def scroll_down(self):
                    return False

            cfg = load_config()
            runner = PracticeRunner(
                self.ocr, cfg, mouse, screen, FakeLLM(), state, _Nav(layout)
            )
            result = runner.run(max_tests=1)
        self.assertEqual(result["completed_skipped"], 1)
        self.assertFalse(
            any(len(c) >= 3 and "平时测试" in str(c[2]) for c in mouse.clicks),
            "已完成的平时测试不能被点击",
        )

    def test_completed_row_without_chapter(self):
        image = load("practice_test_row_completed.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=0, panel_top=0)
        entries = PracticeScanner(self.ocr, self.cfg).scan(image, layout)
        self.assertTrue(entries)
        self.assertTrue(entries[0].completed)

    def test_result_page_detected_submitted(self):
        page, image, _mouse = self._page("practice_result_page.png")
        lines = page._read(image)
        text = page._text(lines)
        self.assertTrue(page._is_submitted_text(text), "得分/解析结果页应判定为已提交")
        self.assertIsNone(page._find_save(lines))
        self.assertTrue(page.wait_for_page(timeout=2.0))
        result = page.run(close_tab=False)
        self.assertTrue(result.get("submitted"))

    def test_close_result_tab(self):
        page, _image, mouse = self._page("practice_result_page.png")
        page._active_window_title = lambda: ""
        page.close_tab()
        self.assertTrue(
            any(c and c[0] == "hotkey" for c in mouse.clicks),
            "结果页没有暂存/提交按钮，也必须关闭标签页",
        )

    def test_auto_closed_to_course_does_not_close_course(self):
        page, _image, mouse = self._page("page_full.png")
        page._active_window_title = lambda: "课程学习_智慧树 - Google Chrome"
        page.close_tab()
        self.assertFalse(
            any(c and c[0] == "hotkey" for c in mouse.clicks),
            "平台自动关考试标签后，绝不能再 Ctrl+W 关闭课程学习标签",
        )

    def test_white_page_not_ready(self):
        image = np.full((600, 800, 3), 255, dtype=np.uint8)
        page, _image, _mouse = self._page_image(image)
        self.assertFalse(page.wait_for_page(timeout=0.2), "白屏不能被当成练习完成")

    def test_practice_scanner_finds_test_row(self):
        image = load("page_full.png")
        layout = Layout.manual(image.shape[1], image.shape[0], panel_left=1017, panel_top=232)
        entries = PracticeScanner(self.ocr, self.cfg).scan(image, layout)
        self.assertTrue(entries, "页面里应识别到‘平时测试’行")
        self.assertTrue(any("平时测试" in e.title for e in entries))
        self.assertTrue(any(e.chapter == "第一章" for e in entries))

    def test_replica_practice_page_fields(self):
        page, image, _mouse = self._page("exam_practice_replica.png")
        lines = page._read(image)
        text = page._text(lines)
        self.assertEqual(page._question_count(lines, text), 5)
        self.assertEqual(page._chapter(text), "第一章")
        self.assertIsNotNone(page._find_save(lines))
        self.assertIsNotNone(page._find_submit(lines))
        q = page.read_question(image)
        self.assertIsNotNone(q)
        self.assertEqual(sorted(q.options), ["A", "B", "C", "D"])
        self.assertFalse(any(page._option_selected(image, r) for r in q.option_rects.values()))
        numbers = page._answer_card_numbers(image, 5)
        self.assertTrue(all(i in numbers for i in range(1, 6)), "答题卡数字 1/5 应从相邻数字补全")

    def test_linear_practice_options_letters(self):
        page, image, _mouse = self._page("practice_linear.png")
        lines = page._read(image)
        self.assertEqual(page._question_count(lines, page._text(lines)), 10)
        q = page.read_question(image)
        self.assertIsNotNone(q)
        self.assertEqual(sorted(q.options), ["A", "B", "C", "D"])

    def test_selected_a_detected_only(self):
        page, image, _mouse = self._page("practice_history.png")
        q = page.read_question(image)
        self.assertIsNotNone(q)
        selected = [k for k, r in q.option_rects.items() if page._option_selected(image, r)]
        self.assertEqual(selected, ["A"])
        lines = page._read(image)
        self.assertIsNone(page._find_save(lines), "没有暂存按钮的页面不应被当作可作答练习")

    def test_stash_modal_10q_confirmed_once(self):
        page, image, mouse = self._page("practice_stash_modal_10q.png")
        lines = page._read(image)
        self.assertEqual(page._modal_kind(page._text(lines)), "stash")
        self.assertTrue(page._confirm_stash())
        labels = [str(c[2]) for c in mouse.clicks if len(c) >= 3]
        self.assertIn("确定暂存", labels)
        self.assertFalse(any("提交" in x or "取消" in x for x in labels), labels)

    def test_close_course_page_leaves_tab(self):
        page, _image, mouse = self._page("page_full.png")
        page._active_window_title = lambda: "课程学习_智慧树 - Google Chrome"
        self.assertTrue(page.close_tab())
        self.assertFalse(
            any(c and c[0] == "hotkey" for c in mouse.clicks),
            "平台自动关标签后已回到课程页，不能再按 Ctrl+W 关掉课程",
        )

    def test_stash_modal_confirmed_once(self):
        page, image, mouse = self._page("practice_stash_modal.jpg")
        lines = page._read(image)
        self.assertEqual(page._modal_kind(page._text(lines)), "stash")
        self.assertTrue(page._confirm_stash())
        labels = [str(c[2]) for c in mouse.clicks if len(c) >= 3]
        self.assertIn("确定暂存", labels)
        self.assertFalse(any("提交" in x for x in labels), labels)

    def test_submit_modal_is_never_confirmed(self):
        self.assertEqual(PracticePage._modal_kind("提示\n确认提交本次作业吗\n取消\n确认提交"), "submit")
        self.assertEqual(PracticePage._modal_kind("提示\n你确定要暂存作业吗\n取消\n确定"), "stash")
        self.assertIsNone(PracticePage._modal_kind("答题后请点【下一题】保存答案，全部试题答完后请点击右上角【提交】按钮"))

    def test_practice_run_never_clicks_submit(self):
        llm = FakeLLM(["D"])
        page, _image, mouse = self._page("exam_practice_replica.png", llm)
        # 静态截图无法真正翻页；模拟翻页成功，专门验证全程不提交/最后暂存
        page.next_question = lambda count, index: True
        result = page.run(close_tab=False)
        self.assertTrue(result["opened"])
        self.assertEqual(result["count"], 5)
        self.assertTrue(result["saved"])
        self.assertEqual(llm.calls, 5)
        labels = [str(c[2]) for c in mouse.clicks if len(c) >= 3]
        self.assertFalse(any("提交" in x for x in labels), labels)
        self.assertFalse(
            any(x.startswith("第 ") and x.endswith(" 题") for x in labels),
            f"不应再点击答题卡数字跳题: {labels}",
        )
        self.assertIn("暂存作业", labels)


class FailingLLM:
    def answer(self, question, image_bgr=None, **kwargs):
        raise LLMError("大模型返回无法解析: 空内容")


class FakeLLM:
    def __init__(self, letters=None):
        self.letters = letters or ["D"]
        self.calls = 0

    def answer(self, question, image_bgr=None, **kwargs):
        self.calls += 1
        valid = [x for x in self.letters if x in question.options] or [sorted(question.options)[0]]
        return LLMAnswer(letters=valid, confidence=0.9, reason="test", source="image")
