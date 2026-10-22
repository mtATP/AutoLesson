from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import vision
from .config import load_config
from .controller import AutoLesson
from .layout import Layout
from .llm import LLMClient
from .ocr import OcrEngine
from .panel import PanelScanner, parse_duration
from .player import PlayerMonitor, parse_clock, parse_time_range
from .quiz import parse_feedback, parse_quiz
from .types import STATUS_COMPLETED, STATUS_IN_PROGRESS, STATUS_NOT_STARTED, Rect

log = logging.getLogger("autolesson.selftest")


class Results:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def check(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            self.items.append((name, True, ""))
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            self.items.append((name, False, str(exc)))
            print(f"  [FAIL] {name}: {exc}")

    def summary(self) -> int:
        failed = [x for x in self.items if not x[1]]
        print("\n" + "=" * 72)
        print(f"自检结果: {len(self.items) - len(failed)}/{len(self.items)} 通过")
        for name, ok, detail in self.items:
            if not ok:
                print(f"  FAIL {name}: {detail}")
        return 1 if failed else 0


def _draw_rect(img: np.ndarray, rect: Rect, color: tuple[int, int, int], label: str = "") -> None:
    cv2.rectangle(img, (rect.x0, rect.y0), (rect.x1, rect.y1), color, 2)
    if label:
        y = max(14, rect.y0 - 5)
        cv2.putText(img, label, (rect.x0 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _status_color(status: str) -> tuple[int, int, int]:
    return {
        STATUS_COMPLETED: (0, 160, 0),
        STATUS_IN_PROGRESS: (0, 120, 255),
        STATUS_NOT_STARTED: (80, 80, 80),
    }.get(status, (0, 0, 0))


def _load(path: Path, required: bool = True) -> Optional[np.ndarray]:
    image = cv2.imread(str(path))
    if image is None and required:
        raise FileNotFoundError(f"无法读取测试图: {path}")
    return image


def run_selftest(fixtures: Path, out_dir: Path) -> int:
    logging.basicConfig(level=logging.WARNING)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("AutoLesson 离线自检")
    print(f"截图目录: {fixtures}")
    print(f"标注输出: {out_dir}")
    print("=" * 72)

    cfg = load_config()
    ocr = OcrEngine()
    ocr.warmup()
    scanner = PanelScanner(ocr, cfg)
    player = PlayerMonitor(ocr, cfg)
    results = Results()

    # ---------------------------------------------------------------- 纯函数
    print("\n[1] 解析函数")
    results.check(
        "时间解析",
        lambda: _expect(parse_clock("11:56") == 716 and parse_clock("1:02:03") == 3723, "parse_clock 错误"),
    )
    results.check(
        "时间区间解析",
        lambda: _expect(parse_time_range("00:00:15/00:11:56") == (15, 716), "parse_time_range 错误"),
    )
    results.check(
        "时长解析",
        lambda: _expect(parse_duration("● 00:19:36") == 1176, "parse_duration 错误"),
    )
    results.check("判分解析(正确)", lambda: _expect(parse_feedback("正确答案: C")[0] == ["C"], "正确答案解析错误"))
    results.check(
        "判分解析(错误)",
        lambda: _expect(parse_feedback("回答错误 正确答案：B") == (["B"], False), "错误判分解析错误"),
    )
    results.check(
        "大模型 JSON 解析",
        lambda: _expect(
            LLMClient._parse('{"answers":["A","C"],"confidence":0.8,"reason":"测试"}', _dummy_question())[:2]
            == (["A", "C"], 0.8),
            "_parse JSON 错误",
        ),
    )
    results.check(
        "大模型兜底解析",
        lambda: _expect(LLMClient._parse("答案：B", _dummy_question())[0] == ["B"], "_parse 兜底错误"),
    )
    results.check(
        "弹窗分类",
        lambda: _expect(
            AutoLesson._classify_overlay("弹题测验 关闭") == "quiz"
            and AutoLesson._classify_overlay("学前必读 感谢你选择修读") == "must_read"
            and AutoLesson._classify_overlay("同学 每日6点更新") == "daily"
            and AutoLesson._classify_overlay("扫码登录") == "login"
            and AutoLesson._classify_overlay("人脸识别") == "verify",
            "弹窗分类错误",
        ),
    )

    # ---------------------------------------------------------------- 目录
    print("\n[2] 课程目录解析")
    panel1 = _load(fixtures / "panel_chapter1.png")
    layout1 = Layout.manual(panel1.shape[1], panel1.shape[0], panel_left=0, panel_right=panel1.shape[1], panel_top=0)
    lessons1 = scanner.scan(panel1, layout1)

    def check_panel1() -> None:
        statuses = {lesson.index: lesson.status for lesson in lessons1}
        expected = {
            "1.1": STATUS_COMPLETED,
            "1.2": STATUS_IN_PROGRESS,
            "1.3": STATUS_NOT_STARTED,
            "1.4": STATUS_NOT_STARTED,
            "1.5": STATUS_NOT_STARTED,
        }
        _expect(len(lessons1) == 5, f"应识别 5 个视频，实际 {len(lessons1)}: {lessons1}")
        for index, status in expected.items():
            _expect(statuses.get(index) == status, f"{index} 状态应为 {status}，实际 {statuses.get(index)}")
        _expect(all("测试" not in l.title for l in lessons1), "不应包含平时测试")

    results.check("第一章目录与状态", check_panel1)

    panel2 = _load(fixtures / "panel_status.png")
    layout2 = Layout.manual(panel2.shape[1], panel2.shape[0], panel_left=0, panel_right=panel2.shape[1], panel_top=0)
    lessons2 = scanner.scan(panel2, layout2)

    def check_panel2() -> None:
        statuses = {lesson.index: lesson.status for lesson in lessons2}
        expected = {
            "0.1": STATUS_COMPLETED,
            "1.1": STATUS_COMPLETED,
            "1.2": STATUS_COMPLETED,
            "2.1": STATUS_IN_PROGRESS,
            "2.2": STATUS_NOT_STARTED,
        }
        _expect(len(lessons2) == 5, f"应识别 5 个视频，实际 {len(lessons2)}: {lessons2}")
        for index, status in expected.items():
            _expect(statuses.get(index) == status, f"{index} 状态应为 {status}，实际 {statuses.get(index)}")

    results.check("跨章目录与状态", check_panel2)

    page = _load(fixtures / "page_full.png")
    layout_page = Layout.manual(page.shape[1], page.shape[0], panel_left=1017, panel_right=page.shape[1], panel_top=232)
    page_lessons = scanner.scan(page, layout_page)

    def check_page() -> None:
        _expect(len(page_lessons) >= 8, f"完整页面应识别至少 8 个视频，实际 {len(page_lessons)}")
        indexes = {l.index: l for l in page_lessons}
        _expect(indexes.get("1.1") is not None and indexes["1.1"].completed, "1.1 应为已完成")
        _expect(indexes.get("1.2") is not None and indexes["1.2"].status == STATUS_IN_PROGRESS, "1.2 应为学习中")
        # 这些行没有状态图标，且右侧有“AI助教”悬浮球，不能误判
        for index in ("1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4"):
            _expect(
                indexes.get(index) is not None and indexes[index].status == STATUS_NOT_STARTED,
                f"{index} 应为未学习，实际 {indexes.get(index)}",
            )
        _expect(
            indexes["1.2"].percent is not None and 50 <= indexes["1.2"].percent <= 60,
            f"1.2 进度应约为 59%，实际 {indexes['1.2'].percent}",
        )

    results.check("完整页面目录", check_page)

    # ---------------------------------------------------------------- 播放器
    print("\n[3] 播放器状态")
    controls = _load(fixtures / "player_controls.png")
    controls_layout = Layout.manual(
        controls.shape[1], controls.shape[0], panel_left=controls.shape[1], panel_right=controls.shape[1]
    )
    controls_state = player.snapshot(controls, controls_layout)

    def check_controls() -> None:
        _expect(controls_state is not None, "未识别到进度条")
        assert controls_state is not None
        _expect(controls_state.current == 15, f"当前时间应为 15，实际 {controls_state.current}")
        _expect(controls_state.total == 716, f"总时长应为 716，实际 {controls_state.total}")
        _expect(controls_state.playing is True, f"应识别为播放中，实际 {controls_state.playing}")
        _expect(controls_state.play_button is not None, "未识别到播放/暂停按钮")

    results.check("进度条/时间/播放状态", check_controls)

    bar_crop = _load(fixtures / "progress_bar.png")
    padded = np.vstack([np.full((520, bar_crop.shape[1], 3), 20, np.uint8), bar_crop])
    padded_layout = Layout.manual(padded.shape[1], padded.shape[0], panel_left=padded.shape[1], panel_right=padded.shape[1])
    padded_state = player.snapshot(padded, padded_layout)

    def check_padded() -> None:
        _expect(padded_state is not None, "未识别到进度条")
        assert padded_state is not None
        _expect(padded_state.current == 43, f"当前时间应为 43，实际 {padded_state.current}")
        _expect(padded_state.total == 716, f"总时长应为 716，实际 {padded_state.total}")
        _expect(padded_state.ratio is not None and 0.05 < padded_state.ratio < 0.08, f"比例异常 {padded_state.ratio}")

    results.check("进度条像素推算", check_padded)

    results.check("控件隐藏时不误判", lambda: _expect(vision.find_progress_bar(page, 1017) is None, "控件隐藏时不应找到进度条"))

    # ---------------------------------------------------------------- 弹窗
    print("\n[4] 弹窗与弹题")
    quiz_full = _load(fixtures / "quiz_full.png")
    cards = vision.find_cards(quiz_full, 2050, cfg.overlay.card_threshold, cfg.overlay.card_close_kernel)
    results.check("弹题弹窗定位", lambda: _expect(len(cards) >= 1, "未找到弹题弹窗"))

    def check_quiz_full() -> None:
        card = cards[0]
        lines = ocr.read(card.crop(quiz_full), offset=(card.x0, card.y0), upscale=1.4)
        header = "\n".join(ln.text for ln in lines[:3])
        _expect("弹题" in header, f"弹窗标题异常: {header}")
        question = parse_quiz(lines, card, quiz_full)
        _expect(question.kind == "多选题", f"题型应为多选题，实际 {question.kind}")
        _expect(set(question.options) == {"A", "B", "C", "D"}, f"选项异常: {question.options}")
        _expect("弹题测验" not in question.text, "题干不应包含弹窗标题")
        _expect(not question.needs_vision, "纯文字题不应强制视觉模式")

    results.check("多选弹题解析", check_quiz_full)

    quiz_multi = _load(fixtures / "quiz_multi.png")
    cards3 = vision.find_cards(quiz_multi, 10000, cfg.overlay.card_threshold, cfg.overlay.card_close_kernel)

    def check_quiz_multi() -> None:
        card = cards3[0]
        lines = ocr.read(card.crop(quiz_multi), offset=(card.x0, card.y0), upscale=1.4)
        question = parse_quiz(lines, card, quiz_multi)
        _expect(question.kind == "多选题", f"题型错误: {question.kind}")
        _expect(len(question.options) == 4, f"应解析 4 个选项，实际 {question.options}")
        _expect("资政新篇" in question.text, f"题干错误: {question.text}")

    results.check("多选弹题(第二张)", check_quiz_multi)

    quiz_math = _load(fixtures / "quiz_math.png")
    cards4 = vision.find_cards(quiz_math, 1017, cfg.overlay.card_threshold, cfg.overlay.card_close_kernel)

    def check_quiz_math() -> None:
        card = cards4[0]
        lines = ocr.read(card.crop(quiz_math), offset=(card.x0, card.y0), upscale=2.0)
        question = parse_quiz(lines, card, quiz_math)
        _expect(question.needs_vision, "数学题应标记为需要视觉模型")
        _expect(len(question.options) >= 1, "应至少解析出选项 A")

    results.check("数学弹题视觉降级", check_quiz_math)

    quiz_answered = _load(fixtures / "quiz_answered.png")
    cards5 = vision.find_cards(quiz_answered, 10000, cfg.overlay.card_threshold, cfg.overlay.card_close_kernel)

    def check_answered() -> None:
        card = cards5[0]
        lines = ocr.read(card.crop(quiz_answered), offset=(card.x0, card.y0), upscale=1.4)
        text = "\n".join(ln.text for ln in lines)
        letters, correct = parse_feedback(text)
        _expect(letters == ["C"], f"正确答案应为 C，实际 {letters}")
        _expect(correct is True, f"应识别为已答对，实际 {correct}")

    results.check("已作答弹题识别", check_answered)

    # ---------------------------------------------------------------- 公告
    print("\n[5] 公告弹窗")
    for name, expected in (("popup_mustread", "must_read"), ("popup_daily", "daily")):
        image = _load(fixtures / f"{name}.png")
        card_list = vision.find_cards(image, 10000, cfg.overlay.card_threshold, cfg.overlay.card_close_kernel)

        def check_popup(image=image, name=name, expected=expected, card_list=card_list) -> None:
            _expect(len(card_list) >= 1, f"{name} 未找到弹窗")
            card = card_list[0]
            header = ocr.read_text(
                Rect(card.x0, card.y0, card.x1, card.y0 + max(90, int(0.28 * card.height))).crop(image),
                offset=(card.x0, card.y0),
                upscale=1.3,
            )
            kind = AutoLesson._classify_overlay(header)
            _expect(kind == expected, f"应分类为 {expected}，实际 {kind}: {header!r}")

        results.check(f"公告分类 {name}", check_popup)

    # ---------------------------------------------------------------- 标注输出
    print("\n[6] 生成标注图")
    try:
        annotated = page.copy()
        _draw_rect(annotated, layout_page.panel, (0, 140, 255), "course panel")
        for lesson in page_lessons:
            _draw_rect(annotated, lesson.title_rect, _status_color(lesson.status), f"{lesson.index} {lesson.status}")
        cv2.imwrite(str(out_dir / "page_full_annotated.png"), annotated)
        panel_annotated = panel2.copy()
        for lesson in lessons2:
            _draw_rect(panel_annotated, lesson.title_rect, _status_color(lesson.status), f"{lesson.index} {lesson.status}")
        cv2.imwrite(str(out_dir / "panel_status_annotated.png"), panel_annotated)
        quiz_annotated = quiz_multi.copy()
        card = cards3[0]
        _draw_rect(quiz_annotated, card, (0, 180, 0), "quiz dialog")
        lines = ocr.read(card.crop(quiz_multi), offset=(card.x0, card.y0), upscale=1.4)
        question = parse_quiz(lines, card, quiz_multi)
        for letter, rect in question.option_rects.items():
            _draw_rect(quiz_annotated, rect, (255, 0, 0), letter)
        cv2.imwrite(str(out_dir / "quiz_multi_annotated.png"), quiz_annotated)
        player_annotated = controls.copy()
        if controls_state is not None:
            cv2.line(
                player_annotated,
                (0, controls_state.bar_y),
                (controls.shape[1] - 1, controls_state.bar_y),
                (0, 255, 0),
                2,
            )
            if controls_state.play_button:
                _draw_rect(player_annotated, controls_state.play_button, (255, 0, 255), "play/pause")
        cv2.imwrite(str(out_dir / "player_controls_annotated.png"), player_annotated)
        print(f"  已写入 {out_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] 标注图生成失败: {exc}")

    return results.summary()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _dummy_question():
    from .types import QuizQuestion

    return QuizQuestion(
        kind="多选题",
        text="测试题",
        options={"A": "甲", "B": "乙", "C": "丙"},
        option_rects={},
        dialog=Rect(0, 0, 100, 100),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AutoLesson 离线自检")
    parser.add_argument("--fixtures", default="tests/fixtures", help="截图目录")
    parser.add_argument("--out", default="runtime/selftest", help="标注输出目录")
    args = parser.parse_args(argv)
    return run_selftest(Path(args.fixtures), Path(args.out))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
