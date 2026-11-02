"""泛雅章节练习/作业的视觉识别与“只暂存”处理。

安全约定：
- 只点击“暂时保存/暂存”，代码里不存在点击“提交”的路径；
- 提交确认弹窗一律取消；
- 大模型没有有效答案时跳过该题，不随机猜。
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from autolesson.llm import LLMError
from autolesson.ocr import OcrEngine, sort_lines
from autolesson.types import OcrLine, QuizQuestion, Rect

from .layout import FanyaLayout
from .types import FanyaItem

log = logging.getLogger("fanya.practice")

QUESTION_RE = re.compile(
    r"^\s*(\d{1,4})\s*[.．、]?\s*【?\s*(单选题|多选题|判断题)"
)
OPTION_RE = re.compile(r"^\s*([A-H])\s*[、.．:：]\s*(.*)$")
LETTER_RE = re.compile(r"^\s*([A-H])\s*$")
SAVE_RE = re.compile(r"(暂时保存|暂存作业|暂存)")
SUBMIT_RE = re.compile(r"提交")


@dataclass
class FanyaPracticeQuestion:
    number: int
    kind: str
    text: str
    options: dict[str, str] = field(default_factory=dict)
    option_rects: dict[str, Rect] = field(default_factory=dict)
    rect: Rect = Rect(0, 0, 0, 0)

    @property
    def key(self) -> str:
        raw = f"{self.number}|{self.text[:120]}|{sorted(self.options)}"
        return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()


class FanyaPracticeReader:
    def __init__(self, ocr: OcrEngine, upscale: float = 1.4):
        self.ocr = ocr
        self.upscale = float(upscale)

    def read(self, frame: np.ndarray, layout: FanyaLayout) -> list[FanyaPracticeQuestion]:
        rect = self.page_rect(layout)
        lines = self._lines(frame, rect)
        return self.parse(lines, rect, frame, layout)

    def _lines(self, frame: np.ndarray, rect: Rect) -> list[OcrLine]:
        lines = self.ocr.read(
            rect.crop(frame), offset=(rect.x0, rect.y0), upscale=self.upscale
        )
        return sort_lines(lines, row_tolerance=max(6, int(0.008 * rect.width)))

    @staticmethod
    def page_rect(layout: FanyaLayout) -> Rect:
        # 题目标题/工具栏在 panel_top 上方；答题页正文从这里开始
        return Rect(
            layout.content.x0,
            max(layout.content.y0, layout.panel.y0),
            layout.content.x1,
            layout.content.y1,
        )

    def has_practice(self, frame: np.ndarray, layout: FanyaLayout) -> bool:
        rect = self.page_rect(layout)
        lines = self._lines(frame, rect)
        text = "\n".join(ln.text for ln in lines)
        return bool(
            any(QUESTION_RE.match(ln.text.strip()) for ln in lines)
            or "暂时保存" in text
            or "暂存" in text
            or "题量" in text
        )

    def parse(
        self,
        lines: list[OcrLine],
        rect: Rect,
        frame: Optional[np.ndarray] = None,
        layout: Optional[FanyaLayout] = None,
    ) -> list[FanyaPracticeQuestion]:
        starts: list[tuple[int, re.Match[str], OcrLine]] = []
        for i, line in enumerate(lines):
            m = QUESTION_RE.match(line.text.strip())
            if m:
                starts.append((i, m, line))
        if not starts:
            return []
        questions: list[FanyaPracticeQuestion] = []
        for n, (start_i, match, start_line) in enumerate(starts):
            end_i = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
            block = lines[start_i:end_i]
            number = int(match.group(1))
            kind = match.group(2)
            # 题干：起始行去掉题号/题型，加上选项之前的换行
            first_text = start_line.text[match.end():].lstrip("】] ").strip()
            text_parts = [first_text] if first_text else []
            option_rows: dict[str, tuple[OcrLine, str]] = {}
            question_bottom = block[-1].rect.y1 if block else start_line.rect.y1
            for line in block[1:]:
                if SAVE_RE.search(line.text) or SUBMIT_RE.search(line.text):
                    break
                m_opt = OPTION_RE.match(line.text.strip())
                if m_opt:
                    option_rows[m_opt.group(1).upper()] = (line, m_opt.group(2).strip())
                    continue
                m_letter = LETTER_RE.match(line.text.strip())
                if m_letter:
                    letter = m_letter.group(1).upper()
                    text_line = self._near_option_text(line, block)
                    text = text_line.text.strip() if text_line is not None else ""
                    option_rows[letter] = (line, text)
                    continue
                if not option_rows:
                    text_parts.append(line.text.strip())
            options: dict[str, str] = {}
            option_rects: dict[str, Rect] = {}
            for letter, (line, text) in sorted(option_rows.items()):
                options[letter] = text
                r = line.rect
                text_line = self._near_option_text(line, block)
                if text_line is not None and text:
                    r = r.union(text_line.rect)
                option_rects[letter] = r
            question = " ".join(p for p in text_parts if p).strip()
            if not question:
                continue
            questions.append(
                FanyaPracticeQuestion(
                    number=number,
                    kind=kind,
                    text=question,
                    options=options,
                    option_rects=option_rects,
                    rect=Rect(
                        min(ln.rect.x0 for ln in block),
                        start_line.rect.y0,
                        max(ln.rect.x1 for ln in block),
                        question_bottom,
                    ),
                )
            )
        return questions

    @staticmethod
    def _near_option_text(letter: OcrLine, block: list[OcrLine]) -> Optional[OcrLine]:
        candidates = []
        for line in block:
            if line is letter:
                continue
            text = line.text.strip()
            if not text or LETTER_RE.match(text) or OPTION_RE.match(text):
                continue
            if line.rect.x0 < letter.rect.x1 + 2:
                continue
            if abs(line.rect.cy - letter.rect.cy) > max(10, int(0.8 * letter.rect.height)):
                continue
            candidates.append(line)
        if not candidates:
            return None
        return min(candidates, key=lambda ln: abs(ln.rect.cy - letter.rect.cy) + (ln.rect.x0 - letter.rect.x1) * 0.01)

    def save_button(self, lines: list[OcrLine]) -> Optional[OcrLine]:
        candidates = [
            ln for ln in lines
            if SAVE_RE.search(ln.text) and not SUBMIT_RE.search(ln.text)
        ]
        return max(candidates, key=lambda ln: ln.rect.cy) if candidates else None


class FanyaPracticeSolver:
    """在一页/多页章节练习中作答，只点击暂时保存。"""

    def __init__(
        self,
        ocr: OcrEngine,
        cfg,
        mouse,
        screen,
        llm,
        state,
        logger: Any = None,
        stop_event=None,
    ):
        self.ocr = ocr
        self.cfg = cfg
        self.mouse = mouse
        self.screen = screen
        self.llm = llm
        self.state = state
        self.log = logger or log
        self.stop_event = stop_event
        self.reader = FanyaPracticeReader(ocr)

    def _check_stop(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            from autolesson.types import UserAbort

            raise UserAbort("收到停止信号")

    def looks_like(self, frame: np.ndarray, layout: FanyaLayout) -> bool:
        return self.reader.has_practice(frame, layout)

    @staticmethod
    def _valid_question(q: FanyaPracticeQuestion) -> bool:
        if len(q.options) < 2:
            return False
        if q.kind == "判断题":
            texts = {str(t).strip() for t in q.options.values()}
            return len(q.options) == 2 and texts.issubset({"对", "错", "正确", "错误"})
        return True

    def _remeasure_gap(self, frame, rect, expected, collected, headers, active):
        """题号跳跃时放大重识别当前屏，找回被漏掉的题目。"""
        if not expected or not headers:
            return None
        last = max(collected) if collected else 0
        if max(headers) <= last + 1:
            return None
        try:
            reader = FanyaPracticeReader(self.ocr, upscale=2.2)
            lines = reader._lines(frame, rect)
            questions = reader.parse(lines, rect)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("练习高倍率重识别失败: %s", exc)
            return None
        new = [q.number for q in questions if q.number not in collected]
        if not new:
            return None
        self.log.warning("练习题号跳跃，高倍率重识别补回题号=%s", new)
        return lines, questions

    def solve(self, layout: FanyaLayout) -> str:
        """边扫边答：题目完整且答案选项可见时立即选择，不跑第二遍。"""
        self._check_stop()
        settings = getattr(self.cfg, "fanya", None)
        limit = max(20, int(getattr(settings, "practice_max_views", 400)))
        rect = self.reader.page_rect(layout)
        collected: dict[int, FanyaPracticeQuestion] = {}
        boundaries: set[int] = set()
        answers: dict[int, list[str]] = {}
        applied: set[tuple[int, str]] = set()
        expected = None
        active = None
        reached_end = False
        try:
            self._scroll_top(layout)
        except Exception as exc:  # noqa: BLE001
            self.log.error("练习页无法回到顶部: %s", exc)
            return "failed"
        previous = None
        last_page = ""
        unchanged = 0
        fallback_used = False
        for view in range(limit):
            self._check_stop()
            frame = self.screen.grab()
            lines = self.reader._lines(frame, rect)
            count = self._question_count(lines)
            if count is not None:
                expected = max(expected or 0, count)
            headers = [int(m.group(1)) for ln in lines
                       if (m := QUESTION_RE.match(ln.text.strip()))]
            if headers:
                boundaries.update(n for n in collected if n < max(headers))
            live = self._continued_questions(lines, rect, active, collected)
            for q in live:
                old = collected.get(q.number)
                if old is None:
                    collected[q.number] = q
                else:
                    if len(q.text) > len(old.text):
                        old.text = q.text
                    for letter, text in q.options.items():
                        if len(text) >= len(old.options.get(letter, "")):
                            old.options[letter] = text
            if headers:
                active = headers[-1]
            retry = self._remeasure_gap(frame, rect, expected, collected, headers, active)
            if retry is not None:
                lines, questions = retry
                headers = [int(m.group(1)) for ln in lines
                           if (m := QUESTION_RE.match(ln.text.strip()))]
                live = self._continued_questions(lines, rect, active, collected)
                for q in live:
                    old = collected.get(q.number)
                    if old is None:
                        collected[q.number] = q
                    else:
                        for letter, text in q.options.items():
                            if len(text) >= len(old.options.get(letter, "")):
                                old.options[letter] = text
                if headers:
                    active = headers[-1]
            save = self.reader.save_button(lines)
            self.log.info("练习采集第 %d 屏: 题号=%s 累计=%s 题量=%s",
                          view + 1, headers, sorted(collected), expected)
            moved_for_answer = False
            for q in live:
                if q.number in answers and all(
                    (q.number, letter) in applied for letter in answers[q.number]
                ):
                    continue
                if not self._question_complete(q, live, save, rect):
                    continue
                merged = collected.get(q.number, q)
                if not self._valid_question(merged):
                    self.log.error("练习题 %d 题型/选项不一致，拒绝点击: %s %s",
                                   q.number, merged.kind, merged.options)
                    continue
                # 下一题尚未采集到时先不答，避免页面上滚/选项错挂后点错题
                if expected and q.number < expected and (q.number + 1) not in collected:
                    self.log.info("练习题 %d 的下一题尚未采集，暂缓作答", q.number)
                    continue
                if q.number not in answers:
                    letters = self._answer_question(merged, frame)
                    if not letters:
                        self.log.warning("练习题 %d 无有效答案，保留该题待复核", q.number)
                        continue
                    answers[q.number] = letters
                    self.log.info("练习题 %d [%s] 完整选项=%s 答案=%s",
                                  q.number, q.kind, merged.options, letters)
                for letter in answers[q.number]:
                    token = (q.number, letter)
                    if token in applied:
                        continue
                    target = q.option_rects.get(letter)
                    if target is not None and rect.y0 + 20 <= target.y0 and target.y1 <= rect.y1 - 40:
                        self.mouse.click_rect(target, f"练习第 {q.number} 题选择 {letter}")
                        applied.add(token)
                        time.sleep(0.25)
                        continue
                    if self._locate_option(q.number, letter, layout, rect, merged):
                        applied.add(token)
                        moved_for_answer = True
                    else:
                        self.log.warning("练习第 %d 题选项 %s 未定位", q.number, letter)
            if save is not None and not moved_for_answer:
                boundaries.update(collected)
                reached_end = True
                break
            if moved_for_answer:
                self.log.info("答案定位后重新向下滚动回保存区")
                self._scroll_down(layout)
                continue
            page_text = "\n".join(ln.text for ln in lines)
            if page_text == last_page:
                if fallback_used:
                    self.log.error("练习滚动后内容未变化，停止本页")
                    return "failed"
                fallback_used = True
                self.log.info("练习页内容未变化，尝试备用滚动")
                self._scroll_fallback(layout)
                last_page = page_text
                continue
            last_page = page_text
            self.log.info("练习小步向下滚动：6 个方向键")
            self._scroll_down(layout)
        if not reached_end:
            self.log.error("练习采集未到保存区域，不标记完成")
            return "failed"
        if expected is None:
            self.log.error("未识别到题量，无法确认是否漏题，保留页面待复核")
            return "failed"
        missing = sorted(set(range(1, expected + 1)) - set(collected))
        incomplete = [n for n, q in collected.items()
                      if n not in boundaries or not self._valid_question(q)
                      or any(not text for text in q.options.values())]
        if missing or incomplete:
            self.log.error("练习采集不完整：漏题=%s 选项不完整=%s；不保存完成记录",
                           missing, incomplete)
            return "failed"
        wanted = {(n, letter) for n, letters in answers.items() for letter in letters}
        if applied != wanted:
            self.log.error("仍有未定位答案 %s，不标记练习完成", sorted(wanted - applied))
            return "failed"
        return self._click_save(frame, layout, lines, save)

    def _locate_option(
        self, number: int, letter: str, layout: FanyaLayout, rect: Rect,
        question: Optional[FanyaPracticeQuestion] = None,
    ) -> bool:
        """目标选项不在当前屏时，小步上下滚动定位后再点。"""
        cached = {number: question} if question is not None else {
            number: FanyaPracticeQuestion(number, "", "")
        }
        for direction in (0, -1, 1):
            steps = 1 if direction == 0 else 10
            for _ in range(steps):
                lines = self.reader._lines(self.screen.grab(), rect)
                live = self._continued_questions(lines, rect, number, cached)
                target = None
                for q in live:
                    if q.number == number:
                        target = q.option_rects.get(letter)
                        break
                if target is not None and target.y0 >= rect.y0 + 20 and target.y1 <= rect.y1 - 40:
                    self.mouse.click_rect(target, f"练习第 {number} 题选择 {letter}")
                    time.sleep(0.25)
                    return True
                if direction == 0:
                    break
                if direction < 0:
                    for _ in range(6):
                        self.mouse.press("up")
                        time.sleep(0.06)
                else:
                    for _ in range(6):
                        self.mouse.press("down")
                        time.sleep(0.06)
                time.sleep(0.5)
        return False

    def _answer_question(self, q: FanyaPracticeQuestion, frame) -> list[str]:
        if self.llm is None:
            self.log.error("练习未配置大模型")
            return []
        question = QuizQuestion(
            kind=q.kind, text=q.text, options=q.options,
            option_rects=q.option_rects, dialog=q.rect, needs_vision=True,
        )
        try:
            answer = self.llm.answer(
                question, image_bgr=q.rect.crop(frame), auto_switch=True
            )
            letters = list(dict.fromkeys(x for x in answer.letters if x in q.options))
        except Exception as exc:  # noqa: BLE001
            self.log.warning("练习题大模型失败，跳过本题: %s", exc)
            return []
        if q.kind in ("单选题", "判断题") and len(letters) > 1:
            letters = letters[:1]
        # 多选题只答一个时强制再问一次，避免漏选
        if q.kind == "多选题" and len(letters) < 2:
            self.log.warning("多选题只返回 %s，强制重新作答", letters)
            retry = QuizQuestion(
                kind="多选题",
                text="多选题，请选出所有正确选项；" + q.text,
                options=q.options,
                option_rects=q.option_rects,
                dialog=q.rect,
                needs_vision=True,
            )
            try:
                answer2 = self.llm.answer(
                    retry, image_bgr=q.rect.crop(frame), auto_switch=True
                )
                letters2 = list(dict.fromkeys(
                    x for x in answer2.letters if x in q.options
                ))
                if len(letters2) > len(letters):
                    letters = letters2
            except Exception as exc:  # noqa: BLE001
                self.log.warning("多选题二次作答失败: %s", exc)
        return letters

    @staticmethod
    def _question_count(lines: list[OcrLine]) -> Optional[int]:
        for line in lines:
            match = re.search(r"(?:题量|总题数|题目数)\s*[:：]?\s*(\d{1,4})", line.text)
            if match:
                return int(match.group(1))
        section_counts = [int(m.group(1)) for line in lines
                          if (m := re.search(r"共\s*(\d+)\s*题", line.text))]
        return sum(section_counts) if section_counts else None

    def _continued_questions(self, lines, rect, active, collected):
        # 已滚出顶部的题干用缓存题号补回，仅用于把尾部选项归到原题。
        work = list(lines)
        first_header = next((i for i, ln in enumerate(work)
                             if QUESTION_RE.match(ln.text.strip())), len(work))
        if active in collected and any(
            LETTER_RE.match(ln.text.strip()) or OPTION_RE.match(ln.text.strip())
            for ln in work[:first_header]
        ):
            q = collected[active]
            work.insert(0, OcrLine(f"{active}【{q.kind}】{q.text}",
                                  Rect(rect.x0, rect.y0, rect.x0 + 100, rect.y0 + 1)))
        return self.reader.parse(work, rect)

    @staticmethod
    def _question_complete(
        q: FanyaPracticeQuestion,
        questions: list[FanyaPracticeQuestion],
        save: Optional[OcrLine],
        rect: Rect,
    ) -> bool:
        """题目完整才作答：后面有题且最后一个选项离底边足够远，或已到保存按钮。"""
        has_next = any(other.number > q.number for other in questions)
        last_bottom = max(
            (r.y1 for r in q.option_rects.values()), default=q.rect.y1
        )
        near_bottom = last_bottom > rect.y1 - max(80, int(0.14 * rect.height))
        return save is not None or (has_next and not near_bottom)

    def _click_save(
        self, frame: np.ndarray, layout: FanyaLayout, lines: list[OcrLine], save: OcrLine
    ) -> str:
        if SUBMIT_RE.search(save.text):
            self.log.error("拒绝点击提交按钮，练习保持未提交")
            return "failed"
        self.log.info("点击暂时保存（绝不提交）")
        self.mouse.click_rect(save.rect.expand(8), "暂时保存")
        time.sleep(1.5)
        # 只处理“暂存/保存”确认；出现“确认提交”一律取消
        lines2 = self.reader._lines(self.screen.grab(), self.reader.page_rect(layout))
        text = "\n".join(ln.text for ln in lines2)
        if "提交" in text:
            # 页面只要出现“提交”，就不点任何确认按钮；宁可保存失败也不提交。
            cancel = next(
                (ln for ln in lines2 if ln.text.strip() in ("取消", "暂不提交", "关闭")),
                None,
            )
            if cancel is not None:
                self.mouse.click_rect(cancel.rect.expand(8), "取消提交确认")
            self.log.error("检测到提交相关提示，已取消；练习只暂存不提交")
            return "failed"
        if "确定" in text and ("暂存" in text or "保存" in text):
            ok = next((ln for ln in lines2 if ln.text.strip() == "确定"), None)
            if ok is not None:
                self.mouse.click_rect(ok.rect.expand(8), "确定暂存")
                time.sleep(1.0)
        if any(w in text for w in ("保存成功", "暂存成功", "已保存", "作业已保存")):
            self.log.info("练习已暂时保存")
            return "saved"
        self.log.warning("已点击暂时保存但未确认成功，不写练习完成记录")
        return "unconfirmed"

    def _save_practice_index(self, layout: FanyaLayout, question: FanyaPracticeQuestion) -> None:
        # 仅用于调试记录，真正的完成标记由 controller 在 selected 保存后写
        pass

    def _focus(self, layout: FanyaLayout) -> None:
        try:
            self.mouse.click(layout.content.cx, layout.panel.y0 + 30, "聚焦练习页")
        except Exception:  # noqa: BLE001
            pass

    def _scroll_top(self, layout: FanyaLayout) -> None:
        """用 Home 回到练习页顶部；滚轮在 iframe 内不可靠。"""
        self._focus(layout)
        try:
            self.mouse.press("home")
            time.sleep(0.8)
            self.mouse.hotkey("ctrl", "home")
            time.sleep(0.8)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"练习回顶失败: {exc}") from exc

    def _scroll_down(self, layout: FanyaLayout) -> None:
        # Down 方向键在 iframe/正文内都可控；每次 6 行，保留上下文重叠。
        self.log.info("练习小步向下滚动：6 个方向键")
        for _ in range(6):
            self.mouse.press("down")
            time.sleep(0.06)
        time.sleep(0.5)

    def _scroll_fallback(self, layout: FanyaLayout) -> None:
        # 方向键也无效时，尝试拖动正文右侧滚动条
        x = max(0, layout.content.x1 - 10)
        y0 = layout.content.y0 + int(0.35 * layout.content.height)
        y1 = min(layout.content.y1 - 20, y0 + int(0.25 * layout.content.height))
        try:
            self.mouse.drag(x, y0, x, y1, "拖动练习滚动条")
            time.sleep(0.8)
        except Exception:  # noqa: BLE001
            pass
