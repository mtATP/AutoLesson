from __future__ import annotations

import logging
import re
import time

import cv2
from typing import Optional

import numpy as np

from . import vision
from .config import Config
from .llm import LLMConfigError, LLMError, random_fallback, report_llm_fallback
from .ocr import OcrEngine, sort_lines
from .types import FatalError, OcrLine, QuizQuestion, Rect

log = logging.getLogger("autolesson.quiz")

KIND_RE = re.compile(r"【\s*(单选题|多选题|判断题|单选|多选|判断)\s*】?")
# 允许选项前出现已选圆点/勾选符号，例如 "● A. -42" / "✓ B. ..."
OPTION_RE = re.compile(r"^[^\w\u4e00-\u9fff]*([A-H])\s*[.、．:：]\s*(.*)$", re.S)
LETTER_ONLY_RE = re.compile(r"^[^\w\u4e00-\u9fff]*([A-H])\s*[.、．:：]?\s*$")
FEEDBACK_RE = re.compile(
    r"正确答案\s*[:：]?\s*([A-H](?:\s*[,，、/和\s]?\s*[A-H])*)"
)
MATH_WORDS = (
    "行列式",
    "矩阵",
    "计算",
    "求值",
    "的值",
    "化简",
    "积分",
    "微分",
    "极限",
    "概率",
    "方程",
    "根号",
    "向量",
    "函数",
    "公式",
    "_____",
    "____",
)
DESCRIPTION_WORDS = (
    "弹题是为了帮助",
    "不会影响到大家",
    "巩固知识点",
)
HEADER_WORDS = (
    "弹题测验",
    "AI助教小智",
    "学习过程中有问题",
)
FOOTER_WORDS = ("关闭", "提交", "确定", "下一题", "上一题", "正确答案", "回答正确", "回答错误")
STRAY_WORDS = ("正确", "错误", "×", "✕", "›", "‹", "<", ">", "|")


def _join_text(parts: list[str]) -> str:
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and out[-1].isascii() and out[-1].isalnum() and part[0].isascii() and part[0].isalnum():
            out += " " + part
        else:
            out += part
    return out.strip()


def _is_stray(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t in STRAY_WORDS:
        return True
    if re.fullmatch(r"[\d\s<>‹›×✕|]+", t):
        return True
    return False


def parse_feedback(text: str) -> tuple[list[str], Optional[bool]]:
    """解析“正确答案：C”和“正确/错误”提示。返回 (正确答案字母, 是否答对)。"""
    letters: list[str] = []
    m = FEEDBACK_RE.search(text or "")
    if m:
        letters = re.findall(r"[A-H]", m.group(1))
    wrong = "错误" in (text or "")
    correct = ("正确" in (text or "")) and not wrong
    if not letters and not wrong and not correct:
        return [], None
    return letters, (False if wrong else True if correct else None)


def is_feedback_text(text: str) -> bool:
    """判断弹窗里是否已经出现判分结果。

    平台的结果徽标常和题型粘在同一行，例如“正确 【单选题】”，
    不能只匹配完全等于“正确”的行；但也要排除“以下哪项回答正确”这种题干。
    """
    letters, _correct = parse_feedback(text)
    if letters:
        return True
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^(回答)?正确(?:$|[\s【])", line) and "不正确" not in line:
            return True
        if re.match(r"^(回答)?错误(?:$|[\s【])", line):
            return True
    return False


def parse_quiz(
    lines: list[OcrLine],
    dialog: Rect,
    frame: Optional[np.ndarray] = None,
    answer_mode: str = "auto",
) -> QuizQuestion:
    inside = [ln for ln in lines if dialog.contains(ln.cx, ln.cy)]
    inside = sort_lines(inside, row_tolerance=max(6, int(0.012 * dialog.width)))
    if not inside:
        return QuizQuestion(kind="", text="", options={}, option_rects={}, dialog=dialog)

    all_text = "\n".join(ln.text for ln in inside)
    kind_match = KIND_RE.search(all_text)
    kind = _normalize_kind(kind_match.group(1)) if kind_match else ""

    # 选项：先识别 A./B. 行，再合并同行的分割 token 与下一行折行
    options: dict[str, str] = {}
    rects: dict[str, Rect] = {}
    option_lines: list[tuple[OcrLine, str, str]] = []
    for line in inside:
        m = OPTION_RE.match(line.text)
        if not m:
            continue
        letter = m.group(1)
        text = m.group(2).strip()
        rect = line.rect
        if not text:
            # 选项字母与文字被 OCR 拆成两个 token
            candidates = [
                other
                for other in inside
                if other is not line
                and abs(other.cy - line.cy) <= max(6, int(0.7 * line.rect.height))
                and other.rect.cx > line.rect.cx
                and other.rect.x0 - line.rect.x1 < 6 * max(1, line.rect.height)
            ]
            if candidates:
                other = min(candidates, key=lambda ln: ln.rect.x0)
                text = other.text.strip()
                rect = rect.union(other.rect)
        option_lines.append((line, letter, text))
        if text:
            if letter not in options or len(text) > len(options[letter]):
                options[letter] = text
                rects[letter] = rect
            elif letter in rects:
                rects[letter] = rects[letter].union(rect)

    option_y = min((ln.cy for ln, _, _ in option_lines), default=None)
    option_x = min((rects[k].x0 for k in rects), default=dialog.x0 + int(0.1 * dialog.width))
    font_h = int(np.median([ln.rect.height for ln, _, _ in option_lines])) if option_lines else 16

    # 选项文字折行：x 对齐且紧跟上一选项
    if option_lines:
        for line in inside:
            if option_y is None or line.cy <= option_y:
                continue
            if OPTION_RE.match(line.text):
                continue
            if line.rect.x0 < option_x - 1.5 * font_h:
                continue
            if line.rect.cx < dialog.x0 + 0.1 * dialog.width:
                continue
            if any(word in line.text for word in FOOTER_WORDS) or _is_stray(line.text):
                continue
            previous = None
            for letter, rect in rects.items():
                if abs(rect.y1 - line.rect.y0) <= 1.6 * font_h and line.rect.x0 >= rect.x0 - 2:
                    if previous is None or rect.y1 > rects[previous].y1:
                        previous = letter
            if previous is not None:
                options[previous] = _join_text([options[previous], line.text])
                rects[previous] = rects[previous].union(line.rect)

    # 题干：选项之前的非页眉/页脚行
    question_parts: list[str] = []
    for line in inside:
        text = line.text.strip()
        if not text:
            continue
        if option_y is not None and line.cy >= option_y - 0.5 * font_h:
            continue
        if any(word in text for word in DESCRIPTION_WORDS + HEADER_WORDS + FOOTER_WORDS):
            continue
        if _is_stray(text):
            continue
        if KIND_RE.search(text):
            text = KIND_RE.sub("", text).strip()
            if not text:
                continue
        question_parts.append(text)
    question = _join_text(question_parts)
    if not question:
        # 某些弹窗没有题型标记，题干可能紧贴选项
        question = _join_text(
            [
                ln.text
                for ln in inside
                if option_y is None or ln.cy < option_y - 0.5 * font_h
                if not any(w in ln.text for w in DESCRIPTION_WORDS + HEADER_WORDS + FOOTER_WORDS)
            ]
        )

    scores = [ln.score for ln in inside if not _is_stray(ln.text)]
    avg_score = float(np.mean(scores)) if scores else 0.0
    min_opt = min((len(v) for v in options.values()), default=0)
    needs_vision = len(options) < 2 or min_opt < 1
    if any(word in question for word in MATH_WORDS) and len(re.sub(r"\W", "", question)) < 40:
        needs_vision = True
    if avg_score < 0.55:
        needs_vision = True
    if answer_mode == "image":
        needs_vision = True
    elif answer_mode == "text":
        needs_vision = False

    return QuizQuestion(
        kind=kind,
        text=question,
        options=options,
        option_rects=rects,
        dialog=dialog,
        header=_join_text([ln.text for ln in inside[:3]]),
        needs_vision=needs_vision,
        ocr_score=avg_score,
        raw_lines=inside,
    )


def _normalize_kind(text: str) -> str:
    if "多" in text:
        return "多选题"
    if "判断" in text:
        return "判断题"
    return "单选题"


def find_close_button(lines: list[OcrLine], dialog: Rect) -> Optional[Rect]:
    candidates = [
        ln
        for ln in lines
        if "关闭" in ln.text
        and "未作答" not in ln.text
        and "不能关闭" not in ln.text
        and ln.cy > dialog.y0 + 0.5 * dialog.height
    ]
    if not candidates:
        return None
    line = max(candidates, key=lambda ln: ln.cy)
    return line.rect.expand(8)


def find_x_button(lines: list[OcrLine], dialog: Rect) -> Optional[Rect]:
    # 弹窗可能与白色视频画面粘连，卡片检测范围会偏大；在顶部 30% 内找 X 更稳
    top = Rect(dialog.x0, dialog.y0, dialog.x1, dialog.y0 + int(0.30 * dialog.height))
    for line in lines:
        text = line.text.strip()
        if text in ("×", "✕", "x", "X", "╳") and top.contains(line.cx, line.cy):
            return line.rect.expand(8)
    return None


def find_submit_button(lines: list[OcrLine], dialog: Rect) -> Optional[Rect]:
    for line in lines:
        if "提交" in line.text and line.cy > dialog.y0 + 0.5 * dialog.height:
            return line.rect.expand(8)
    return None


class QuizHandler:
    """处理弹题：滚动读取完整题目、调用大模型、点选、关闭。"""

    def __init__(self, ocr: OcrEngine, cfg: Config, mouse, screen, llm, state, logger=None, notifier=None):
        self.ocr = ocr
        self.cfg = cfg
        self.mouse = mouse
        self.screen = screen
        self.llm = llm
        self.state = state
        self.log = logger or log
        self.notifier = notifier
        self._last_stitched: Optional[np.ndarray] = None
        self._last_images: list[np.ndarray] = []

    # ------------------------------------------------------------------ OCR
    def _read_with_lines(self, frame: np.ndarray, dialog: Rect) -> tuple[QuizQuestion, list[OcrLine]]:
        scale = 2.0 if dialog.width < 800 else 1.4
        if dialog.width * scale > 2200:
            scale = 2200.0 / dialog.width
        lines = self.ocr.read(dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=scale)
        return parse_quiz(lines, dialog, frame, str(self.cfg.quiz.answer_mode)), lines

    def read(self, frame: np.ndarray, dialog: Rect) -> QuizQuestion:
        return self._read_with_lines(frame, dialog)[0]

    @staticmethod
    def _expected_letters(q: QuizQuestion) -> list[str]:
        if "判断" in (q.kind or ""):
            return ["A", "B"]
        letters = sorted(k.upper() for k in q.options if k)
        if not letters:
            return []
        # 只补“已出现字母之间”的缺口；A/B/C 就是完整的三选项题，不再强行补 D
        last = max(ord(x) - 65 for x in letters)
        return [chr(65 + i) for i in range(last + 1)]

    @classmethod
    def _complete(cls, q: QuizQuestion, lines: list[OcrLine], dialog: Rect) -> bool:
        """选项数量齐全且关闭按钮在视口内，才跳过滚动。"""
        if q is None or len(q.options) < 1:
            return False
        if len(q.options) < 2:
            return False
        expected = cls._expected_letters(q)
        if any(letter not in q.options for letter in expected):
            return False
        return find_close_button(lines, dialog) is not None

    def _body_point(self, dialog: Rect) -> tuple[int, int]:
        # 含图片的题目正文可能占满中间；把滚轮点放在右侧正文边缘，
        # 避开图片的缩放/拖拽区域，同时不压到右侧滚动条。
        return (
            dialog.x0 + int(0.90 * dialog.width),
            dialog.y0 + int(0.40 * dialog.height),
        )

    def _dialog_signature(self, frame: np.ndarray, dialog: Rect) -> bytes:
        return vision.panel_signature(frame, dialog)

    def _wheel_pixels(self, dialog: Rect, pixels: int, direction: int) -> None:
        """连续发送小步滚轮事件，累计约 pixels 像素。

        Chrome/元素滚动条会截断单次大 delta；含图片的弹题用一个大 scroll(-80)
        往往只动几像素，所以必须拆成 60 像素的小事件。
        """
        remaining = abs(int(pixels))
        x, y = self._body_point(dialog)
        while remaining > 0:
            chunk = min(60, remaining)
            self.mouse.scroll_at(x, y, int(direction) * chunk)
            remaining -= chunk
            time.sleep(0.03)

    def scroll(self, dialog: Rect, pixels: int, wait: Optional[float] = None) -> None:
        self._wheel_pixels(dialog, pixels, -1)
        time.sleep(float(self.cfg.quiz.scroll_wait if wait is None else wait))

    def scroll_to_top(self, dialog: Rect) -> None:
        # 只使用鼠标滚轮：拖滚动条会选中正文文字，导致后续选项点击异常
        step = max(800, int(1.5 * dialog.height))
        for _ in range(6):
            before = self._dialog_signature(self.screen.grab(), dialog)
            self._wheel_pixels(dialog, step, +1)
            time.sleep(0.40)
            after = self._dialog_signature(self.screen.grab(), dialog)
            if vision.signature_distance(before, after) <= 12:
                break

    def _body_image(self, frame: np.ndarray, dialog: Rect) -> np.ndarray:
        """弹窗正文区域；排除顶部标题、底部关闭和右侧滚动条。"""
        w, h = dialog.width, dialog.height
        body = Rect(
            dialog.x0 + int(0.03 * w),
            dialog.y0 + int(0.10 * h),
            dialog.x1 - int(0.06 * w),
            # 底部只留 4%：选项常贴在关闭按钮上方，10% 会把最后一个选项裁掉
            dialog.y1 - int(0.04 * h),
        )
        return body.crop(frame)

    @staticmethod
    def _stitch_vertical(prev: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
        """把滚动后的正文拼接起来；用底部 30% 做模板找重叠区。"""
        if prev is None or prev.size == 0:
            return new.copy()
        if new.size == 0:
            return prev
        tail_h = max(24, min(prev.shape[0] // 2, int(prev.shape[0] * 0.30)))
        tail = prev[-tail_h:, :]
        if tail.shape[1] != new.shape[1] or tail.shape[0] >= new.shape[0]:
            return np.vstack([prev, new])
        try:
            res = cv2.matchTemplate(new, tail, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
        except cv2.error:
            return np.vstack([prev, new])
        if maxv < 0.55:
            return np.vstack([prev, new])
        y = int(maxloc[1])
        rest = new[y + tail_h :, :]
        if rest.size == 0:
            return prev
        return np.vstack([prev, rest])

    def _drag_scrollbar_bottom(self, dialog: Rect) -> None:
        """滚轮不生效时，直接拖动弹窗右侧滚动条到底部。"""
        # 弹窗滚动条在右侧约 3.5% 处；太靠右会落在正文里，拖拽会触发全选文本
        x = dialog.x1 - max(10, int(0.035 * dialog.width))
        y0 = dialog.y0 + int(0.18 * dialog.height)
        y1 = dialog.y1 - int(0.08 * dialog.height)
        try:
            self.mouse.drag(x, y0, x, y1, "弹题滚动到底")
            time.sleep(0.5)
            self.log.info("弹题滚轮无效，已拖动滚动条到底部")
        except Exception as exc:  # noqa: BLE001
            self.log.debug("拖动弹题滚动条失败: %s", exc)

    def _recover_missing_options(self, q: QuizQuestion, dialog: Rect) -> QuizQuestion:
        """C 等选项被公式/图片挤掉时，放大整卡和缺失行再做一次 OCR。"""
        expected = self._expected_letters(q)
        missing = [x for x in expected if x not in q.options]
        if not missing:
            return q
        frame = self.screen.grab()
        # 先整卡高倍 OCR，能补回被低分辨率漏掉的短选项（如 C. -32）
        try:
            lines = self.ocr.read(
                dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=2.2
            )
            page = parse_quiz(lines, dialog, frame, "text")
            for letter in missing:
                if letter in page.options and page.options[letter].strip():
                    q.options[letter] = page.options[letter]
                    q.option_rects[letter] = page.option_rects.get(letter, dialog)
                    self.log.info("补识别选项 %s: %r", letter, page.options[letter][:40])
        except Exception as exc:  # noqa: BLE001
            self.log.debug("整卡补识别选项失败: %s", exc)
        missing = [x for x in expected if x not in q.options]
        if not missing:
            return q
        # 用已知选项行距推算缺失行的 y，再裁一条横带高倍 OCR
        rows = sorted(
            (ord(k.upper()) - 65, r.cy) for k, r in q.option_rects.items() if k
        )
        if rows:
            if len(rows) >= 2:
                idxs = np.array([x[0] for x in rows], dtype=float)
                ys = np.array([x[1] for x in rows], dtype=float)
            else:
                idxs = np.array([rows[0][0]], dtype=float)
                ys = np.array([rows[0][1]], dtype=float)
            for letter in missing:
                target = ord(letter) - 65
                if len(rows) >= 2:
                    y = int(round(float(np.interp(target, idxs, ys))))
                else:
                    step = max(40, int(0.09 * dialog.height))
                    y = int(round(rows[0][1] + (target - rows[0][0]) * step))
                band_h = max(35, int(0.055 * dialog.height))
                band = Rect(
                    dialog.x0 + int(0.04 * dialog.width),
                    max(dialog.y0, y - band_h),
                    dialog.x1 - int(0.04 * dialog.width),
                    min(dialog.y1, y + band_h),
                )
                try:
                    band_lines = self.ocr.read(
                        band.crop(frame), offset=(band.x0, band.y0), upscale=3.0
                    )
                    text = "\n".join(ln.text for ln in band_lines)
                    self.log.info("缺失选项 %s 行 OCR: %r", letter, text[:80])
                    for ln in band_lines:
                        m = OPTION_RE.match(ln.text.strip())
                        if m and m.group(1).upper() == letter:
                            q.options[letter] = m.group(2).strip() or ln.text.strip()
                            q.option_rects[letter] = ln.rect
                            break
                    if letter not in q.options:
                        # OCR 没读到字母时，把这一行最长的文字作为该选项文本
                        body = [
                            ln for ln in band_lines
                            if ln.text.strip() and not any(
                                w in ln.text for w in ("关闭", "上一题", "下一题", "提交")
                            )
                        ]
                        if body:
                            best = max(body, key=lambda ln: len(ln.text.strip()))
                            q.options[letter] = best.text.strip()
                            q.option_rects[letter] = best.rect
                except Exception as exc:  # noqa: BLE001
                    self.log.debug("缺失选项 %s 行 OCR 失败: %s", letter, exc)
        missing = [x for x in expected if x not in q.options]
        if missing:
            # 仍然缺文字时先补占位，避免大模型给出正确字母却被“不在选项”拒绝；
            # 后续按行距点圆点仍可选中。
            for letter in missing:
                q.options[letter] = "（选项图片/文字未识别，按行位置选择）"
            self.log.warning("选项 %s 未识别出文字，已补占位并按行距点选", missing)
        return q

    @staticmethod
    def _append_unique_image(images: list[np.ndarray], image: np.ndarray) -> None:
        if image is None or image.size == 0:
            return
        for old in images:
            if old.shape == image.shape:
                try:
                    if float(np.mean(cv2.absdiff(old, image))) < 2.0:
                        return
                except cv2.error:
                    pass
        images.append(image)

    def collect(self, dialog: Rect) -> QuizQuestion:
        """先读首屏；完整就直接返回，不完整才用滚轮向下滚动并拼接完整长图。"""
        # 正确率优先：不再因为“看到了一个选项+关闭按钮”就跳过滚动。
        # 回到顶部后慢速逐屏扫描，长图题/三选项题都能完整收集。
        self.scroll_to_top(dialog)
        frame = self.screen.grab()
        first, first_lines = self._read_with_lines(frame, dialog)
        if self._complete(first, first_lines, dialog):
            body = self._body_image(frame, dialog)
            self._last_images = [body]
            self._last_stitched = body
            self.log.info("弹题首屏完整，单图提交: %s", sorted(first.options))
            return first
        merged = self.read(frame, dialog)
        seen_sig = self._dialog_signature(frame, dialog)
        best_text = merged.text
        options = dict(merged.options)
        rects = dict(merged.option_rects)
        images: list[np.ndarray] = []
        self._append_unique_image(images, self._body_image(frame, dialog))
        stitched = self._body_image(frame, dialog)
        max_scrolls = max(32, int(self.cfg.quiz.max_scrolls) * 4)
        step = max(180, min(300, abs(int(self.cfg.quiz.scroll_step)) * 45))
        wait = max(0.45, min(0.65, float(self.cfg.quiz.scroll_wait)))
        unchanged = 0
        for _ in range(max_scrolls):
            self.scroll(dialog, step, wait=wait)
            frame = self.screen.grab()
            sig = self._dialog_signature(frame, dialog)
            page = self.read(frame, dialog)
            if len(page.text) > len(best_text):
                best_text = page.text
            if page.kind and not merged.kind:
                merged.kind = page.kind
            merged.needs_vision = merged.needs_vision or page.needs_vision
            for letter, text in page.options.items():
                if letter not in options or len(text) > len(options[letter]):
                    options[letter] = text
                    rects[letter] = page.option_rects.get(letter, rects.get(letter, dialog))
            changed = vision.signature_distance(seen_sig, sig) > 12
            seen_sig = sig
            if changed:
                unchanged = 0
                body = self._body_image(frame, dialog)
                stitched = self._stitch_vertical(stitched, body)
                self._append_unique_image(images, body)
                continue
            unchanged += 1
            if unchanged >= 3:
                break
        self.scroll_to_top(dialog)
        self._last_stitched = stitched
        max_images = max(1, int(getattr(self.cfg.llm, "max_images", 8)))
        if len(images) > max_images:
            # 均匀保留，头和尾必须保留
            idx = np.linspace(0, len(images) - 1, max_images).round().astype(int)
            images = [images[i] for i in sorted(set(idx.tolist()))]
        self._last_images = images
        self.log.info(
            "弹题慢速扫描完成: %d 屏，选项 %s", len(images), sorted(options)
        )
        merged.text = best_text
        merged.options = options
        merged.option_rects = rects
        return self._recover_missing_options(merged, dialog)

    # ------------------------------------------------------------------ 交互
    def is_answered(self, frame: np.ndarray, dialog: Rect) -> tuple[bool, list[str]]:
        lines = self.ocr.read(dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=1.2)
        text = "\n".join(ln.text for ln in lines)
        letters, _correct = parse_feedback(text)
        answered = is_feedback_text(text)
        return answered, letters

    def close(self, frame: np.ndarray, dialog: Rect) -> bool:
        """点击关闭；成功返回 True。"""
        if self._unanswered_open(dialog):
            self.log.warning("检测到未作答提示，先不点外层弹窗的关闭")
            return False
        for attempt in range(3):
            frame = self.screen.grab()
            lines = self.ocr.read(dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=1.2)
            button = find_close_button(lines, dialog)
            if button is not None:
                self.mouse.click_rect(button, "关闭弹题")
            else:
                self.mouse.click(dialog.cx, dialog.y1 - int(0.055 * dialog.height), "关闭弹题(坐标)")
            time.sleep(float(self.cfg.quiz.close_wait))
            if not self._still_open(dialog):
                return True
            if attempt == 0:
                x_btn = find_x_button(lines, dialog)
                if x_btn is not None:
                    self.mouse.click_rect(x_btn, "关闭弹题(X)")
                    time.sleep(0.8)
                    if not self._still_open(dialog):
                        return True
            elif attempt == 1:
                self.mouse.press("esc")
                time.sleep(0.8)
                if not self._still_open(dialog):
                    return True
        return False

    UNANSWERED_WORDS = ("未作答", "不能关闭", "请先作答", "作答后")

    def _dialog_text(self, dialog: Rect) -> str:
        frame = self.screen.grab()
        lines = self.ocr.read(dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=1.2)
        return "\n".join(ln.text for ln in lines)

    def _unanswered_prompt(self, dialog: Rect) -> Optional[Rect]:
        """找到“未作答的弹题不能关闭”的白色小提示卡，而不是外层弹题。"""
        frame = self.screen.grab()
        text = self._dialog_text(dialog)
        if not any(word in text for word in self.UNANSWERED_WORDS):
            return None
        try:
            cards = vision.find_cards(
                frame,
                dialog.x1,
                int(self.cfg.overlay.card_threshold),
                int(self.cfg.overlay.card_close_kernel),
                8,
            )
        except Exception:  # noqa: BLE001
            cards = []
        candidates = []
        for card in cards:
            if not dialog.contains(card.cx, card.cy):
                continue
            if card.width >= 0.92 * dialog.width:
                continue
            prompt_text = self.ocr.read_text(
                card.crop(frame), offset=(card.x0, card.y0), upscale=1.2
            )
            if any(word in prompt_text for word in self.UNANSWERED_WORDS):
                candidates.append(card)
        if candidates:
            # 取最小的白色卡：外层弹题/视频画面更大
            return min(candidates, key=lambda r: r.area)
        # 兜底：用“未作答”文字行估一个提示框，绝不再用外层弹题的 X
        lines = self.ocr.read(dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=1.2)
        hits = [ln for ln in lines if any(w in ln.text for w in self.UNANSWERED_WORDS)]
        if not hits:
            return None
        x0 = min(ln.rect.x0 for ln in hits) - 80
        x1 = max(ln.rect.x1 for ln in hits) + 160
        y0 = min(ln.rect.y0 for ln in hits) - 140
        y1 = max(ln.rect.y1 for ln in hits) + 60
        return Rect(x0, y0, x1, y1).clamp(dialog.x1, dialog.y1)

    def _unanswered_open(self, dialog: Rect) -> bool:
        return self._unanswered_prompt(dialog) is not None

    def _clear_selection(self, dialog: Rect) -> None:
        """点击弹窗空白处/标题，清除拖拽造成的“全选文本”。"""
        try:
            self.mouse.click(
                dialog.x0 + int(0.05 * dialog.width),
                dialog.y0 + int(0.05 * dialog.height),
                "清除文本选择",
            )
            time.sleep(0.2)
        except Exception:  # noqa: BLE001
            pass

    def _dismiss_unanswered(self, dialog: Rect) -> None:
        """只关闭“未作答”小提示，绝不点外层弹题的 X/关闭。"""
        self.log.warning("检测到‘未作答的弹题不能关闭’，关闭提示并重试选项")
        try:
            prompt = self._unanswered_prompt(dialog)
            if prompt is None:
                # 不能按 ESC：ESC 可能直接关掉答题卡，导致后续找不到题目。
                # 用“未作答”文字附近的 X 做兜底；找不到就保持原样。
                frame = self.screen.grab()
                lines = self.ocr.read(
                    dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=1.2
                )
                hits = [
                    ln for ln in lines
                    if any(w in ln.text for w in self.UNANSWERED_WORDS)
                ]
                xs = [
                    ln for ln in lines
                    if ln.text.strip() in ("X", "x", "×", "✕", "╳")
                ]
                if hits and xs:
                    target = max(hits, key=lambda ln: ln.rect.y0)
                    x_btn = min(
                        xs,
                        key=lambda ln: abs(ln.cx - target.rect.x1)
                        + abs(ln.cy - target.rect.y0),
                    )
                    self.mouse.click_rect(x_btn.rect.expand(8), "关闭未作答提示(文字兜底)")
                    time.sleep(0.8)
                else:
                    self.log.warning("未定位到未作答提示卡/X，保持弹题不动，避免 ESC 关掉答题卡")
                return
            frame = self.screen.grab()
            lines = self.ocr.read(
                prompt.crop(frame), offset=(prompt.x0, prompt.y0), upscale=2.0
            )
            x_btn = find_x_button(lines, prompt)
            if x_btn is not None:
                self.mouse.click_rect(x_btn, "关闭未作答提示")
            else:
                # 提示卡的 X 在右上角约 13% 处；用卡片自身坐标，不用外层弹窗
                self.mouse.click(
                    prompt.x1 - int(0.13 * prompt.width),
                    prompt.y0 + int(0.13 * prompt.height),
                    "关闭未作答提示(X)",
                )
            time.sleep(0.8)
            if self._unanswered_open(dialog):
                # 少数皮肤 X 更靠上，第二次用右上角 8% 兜底
                self.mouse.click(
                    prompt.x1 - int(0.08 * prompt.width),
                    prompt.y0 + int(0.08 * prompt.height),
                    "关闭未作答提示(兜底X)",
                )
                time.sleep(0.6)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("关闭未作答提示失败: %s", exc)

    def _force_close(self, dialog: Rect) -> bool:
        """随机兜底判错后的强制关闭：X、Esc、底部关闭坐标依次尝试。"""
        if self._unanswered_open(dialog):
            self.log.warning("检测到未作答提示，拒绝强制关闭外层弹题")
            return False
        try:
            frame = self.screen.grab()
            lines = self.ocr.read(dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=1.2)
            x_btn = find_x_button(lines, dialog)
            if x_btn is not None:
                self.mouse.click_rect(x_btn, "关闭弹题(强制X)")
                time.sleep(0.8)
                if not self._still_open(dialog):
                    return True
            self.mouse.press("esc")
            time.sleep(0.8)
            if not self._still_open(dialog):
                return True
            self.mouse.click(
                dialog.cx,
                dialog.y1 - int(0.06 * dialog.height),
                "关闭弹题(强制坐标)",
            )
            time.sleep(0.8)
            return not self._still_open(dialog)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("强制关闭弹题失败: %s", exc)
            return False

    def _still_open(self, dialog: Rect) -> bool:
        frame = self.screen.grab()
        lines = self.ocr.read(dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=1.2)
        text = "\n".join(ln.text for ln in lines)
        # 弹窗顶部固定显示“弹题测验”；关闭后即使露出含“关闭”字幕的视频也不能误判
        return "弹题" in text or "正确答案" in text

    def _option_rect_current(self, dialog: Rect, letter: str) -> Optional[Rect]:
        frame = self.screen.grab()
        page = self.read(frame, dialog)
        return page.option_rects.get(letter)

    def _scroll_until_option(self, dialog: Rect, letter: str) -> Optional[Rect]:
        # 与 collect 同一套策略：大步长、短等待、滚轮无效时拖滚动条到底
        step = max(300, min(600, abs(int(self.cfg.quiz.scroll_step)) * 90))
        wait = min(0.35, float(self.cfg.quiz.scroll_wait))
        unchanged = 0
        for _ in range(max(16, int(self.cfg.quiz.max_scrolls))):
            rect = self._option_rect_current(dialog, letter)
            if (
                rect is not None
                and rect.y0 > dialog.y0 + 0.10 * dialog.height
                and rect.y1 < dialog.y1 - 0.04 * dialog.height
            ):
                return rect
            before = self.screen.grab()
            sig_before = self._dialog_signature(before, dialog)
            self.scroll(dialog, step, wait=wait)
            after = self.screen.grab()
            sig_after = self._dialog_signature(after, dialog)
            if vision.signature_distance(sig_before, sig_after) <= 12:
                unchanged += 1
                if unchanged >= 2:
                    return rect
            else:
                unchanged = 0
        return None

    def _option_rect_from_layout(
        self, dialog: Rect, letter: str, checkbox_x: int
    ) -> Optional[Rect]:
        """图片选项 OCR 只认出 C/D 时，按已知行距推算 A/B 的圆点行。"""
        try:
            page = self.read(self.screen.grab(), dialog)
        except Exception:  # noqa: BLE001
            return None
        rows = sorted(
            (ord(k.upper()) - 65, r.cy) for k, r in page.option_rects.items() if k
        )
        if not rows:
            return None
        target = ord(letter.upper()) - 65
        if len(rows) >= 2:
            steps = [rows[i + 1][1] - rows[i][1] for i in range(len(rows) - 1)]
            step = float(np.median([x for x in steps if x > 0])) if any(x > 0 for x in steps) else 0.0
        else:
            step = 0.0
        if step <= 8:
            step = max(30.0, 0.09 * dialog.height)
        base_idx, base_y = rows[0]
        y = int(round(base_y + (target - base_idx) * step))
        if not (dialog.y0 < y < dialog.y1):
            return None
        h = max(12, int(step * 0.35))
        return Rect(checkbox_x - h, y - h, checkbox_x + h, y + h)

    def _feedback_visible(self, dialog: Rect) -> bool:
        try:
            frame = self.screen.grab()
            lines = self.ocr.read(
                dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=1.2
            )
            text = "\n".join(ln.text for ln in lines)
            return is_feedback_text(text)
        except Exception:  # noqa: BLE001
            return False

    def _selected(
        self, frame: np.ndarray, rect: Rect, checkbox_x: Optional[int] = None
    ) -> bool:
        """只看选项左侧圆点，不能把含蓝色公式的选项图片当成已选中。"""
        font_h = max(10, rect.height)
        x = checkbox_x if checkbox_x is not None else rect.x0 - int(2.0 * font_h)
        crop = Rect(
            x - int(1.4 * font_h),
            rect.cy - int(1.0 * font_h),
            x + int(1.4 * font_h),
            rect.cy + int(1.0 * font_h),
        ).crop(frame)
        if crop.size == 0:
            return False
        mask = vision.blue_mask(crop)
        return float(mask.mean()) > 0.03

    def apply_answer(self, dialog: Rect, letters: list[str], single: bool) -> bool:
        self.scroll_to_top(dialog)
        # 选项从上到下按字母排列；排序后可单调向下滚动查找
        targets = sorted(letters[:1] if single else letters)
        # 单选圆点位于弹窗左侧固定列；不能每行用文字 x 推算，图片题会偏到矩阵上
        checkbox_x = dialog.x0 + int(0.17 * dialog.width)
        selected_rects: dict[str, Rect] = {}
        for letter in targets:
            rect = self._scroll_until_option(dialog, letter)
            if rect is None:
                rect = self._option_rect_from_layout(dialog, letter, checkbox_x)
                if rect is not None:
                    self.log.info("选项 %s 文字/图片未被 OCR 读到，按行距定位圆点", letter)
            if rect is None:
                self.log.error("未能在弹窗中找到选项 %s", letter)
                return False
            selected_rects[letter] = rect
            y = rect.cy
            frame = self.screen.grab()
            if not self._selected(frame, rect, checkbox_x):
                self._clear_selection(dialog)
                self.mouse.click(checkbox_x, y, f"选择 {letter}")
                time.sleep(0.4)
                frame = self.screen.grab()
                # 单选可能立即判分并导致选项行下移，重新定位后再兜底点击
                current = self._option_rect_current(dialog, letter) or rect
                if not self._selected(frame, current, checkbox_x):
                    self.mouse.click(current.cx, current.cy, f"选择 {letter}(文字)")
                    time.sleep(0.35)
        # 平台判分可能比选项圆点变蓝晚 1 秒；先等一会儿，再看“正确/错误”徽标
        time.sleep(0.8)
        if self._feedback_visible(dialog):
            return True
        # 校验
        self.scroll_to_top(dialog)
        frame = self.screen.grab()
        page = self.read(frame, dialog)
        missing = []
        for letter in targets:
            rect = page.option_rects.get(letter) or selected_rects.get(letter)
            if rect is None:
                rect = self._scroll_until_option(dialog, letter)
            if rect is None or not self._selected(self.screen.grab(), rect, checkbox_x):
                missing.append(letter)
        if missing:
            if self._feedback_visible(dialog):
                return True
            self.log.warning("选项 %s 可能未被选中", missing)
            return False
        return True

    # ------------------------------------------------------------------ LLM
    def _ask_question(self, question: QuizQuestion, body_image: np.ndarray):
        """练习页同款多级降级；视频弹题默认随机兜底，不中断观看。"""
        if not self.cfg.llm_ready():
            answer = random_fallback(question)
            if answer is None:
                raise FatalError("弹题无可用选项", "无 LLM 且选项为空，无法随机兜底")
            report_llm_fallback(
                self.cfg,
                self.screen,
                question.dialog,
                question,
                answer.letters[0],
                LLMError("未配置 LLM，按无 LLM 模式随机选择"),
                self.notifier,
                self.log,
            )
            return answer

        images = [x for x in (self._last_images or []) if x is not None and x.size]
        stitched = self._last_stitched
        if stitched is None or stitched.size == 0:
            stitched = body_image
        if not images:
            images = [stitched]

        if question.needs_vision:
            modes = [
                ("image", "拼接图"),
                ("image", "拼接图重试"),
                ("multi", "多图"),
                ("text", "纯文本"),
            ]
        else:
            modes = [
                ("text", "纯文本"),
                ("image", "拼接图"),
                ("multi", "多图"),
            ]

        answer = None
        last_error: Optional[Exception] = None
        for kind, label in modes:
            try:
                if kind == "image":
                    answer = self.llm.answer(
                        question, images=[stitched], auto_switch=False
                    )
                elif kind == "multi":
                    answer = self.llm.answer(question, images=images, auto_switch=False)
                else:
                    answer = self.llm.answer(question, image_bgr=None, auto_switch=False)
                if answer is not None and answer.letters:
                    self.log.info("弹题 LLM 模式: %s", label)
                    break
                last_error = LLMError("大模型返回空答案")
            except LLMError as exc:
                last_error = exc
                self.log.warning("弹题 LLM %s 失败: %s", label, exc)
            answer = None

        if answer is None or not answer.letters:
            if bool(getattr(self.cfg.llm, "fallback_random_enabled", True)):
                fallback = random_fallback(question)
                if fallback is not None:
                    report_llm_fallback(
                        self.cfg,
                        self.screen,
                        question.dialog,
                        question,
                        fallback.letters[0],
                        last_error or LLMError("大模型未返回有效答案"),
                        self.notifier,
                        self.log,
                    )
                    return fallback
            raise LLMError(
                f"大模型无有效答案，随机兜底不可用或已关闭: {last_error or '空答案'}"
            )
        return answer

    # ------------------------------------------------------------------ 主流程
    def solve(self, frame: np.ndarray, dialog: Rect) -> dict:
        self.log.info("检测到弹题弹窗 %s", dialog)
        answered, correct_letters = self.is_answered(frame, dialog)
        if answered:
            self.log.info("弹题已作答（正确答案: %s），直接关闭", correct_letters or "未知")
            if not self.close(frame, dialog):
                raise FatalError("弹题弹窗无法关闭", "已尝试关闭按钮、右上角 X 和 ESC")
            return {"answered": True, "closed": True, "correct": None, "letters": correct_letters}

        question = self.collect(dialog)
        frame = self.screen.grab()
        self.log.info(
            "题目: %s | 选项: %s | 需视觉: %s",
            question.text[:80],
            {k: v[:24] for k, v in question.options.items()},
            question.needs_vision,
        )
        if len(question.options) < 1:
            self.log.error("题目 OCR 结果不完整: %s", question.prompt_text())
            raise FatalError(
                "弹题识别失败",
                "没有任何可见选项，无法作答或随机兜底，已停止以避免乱点",
            )
        if not question.text:
            self.log.warning("题干 OCR 为空，继续用选项/图像作答: %s", question.prompt_text())

        body_image = self._body_image(frame, dialog)
        try:
            answer = self._ask_question(question, body_image)
        except LLMConfigError:
            raise
        except LLMError as exc:
            raise FatalError("大模型答题失败", str(exc)) from exc
        self.log.info(
            "大模型答案: %s (置信度 %.2f, 来源 %s) %s",
            answer.letters,
            answer.confidence,
            answer.source,
            answer.reason[:80],
        )
        if not answer.letters:
            raise FatalError("大模型未给出有效答案", answer.raw[:200])

        single = question.kind != "多选题" and len(answer.letters) == 1
        applied = self.apply_answer(dialog, answer.letters, single=single)
        if not applied or self._unanswered_open(dialog):
            # 拖拽全选/点击落空时平台会提示“未作答的弹题不能关闭”；
            # 关掉提示、清除选择后重试一次选项，避免直接中断。
            self._dismiss_unanswered(dialog)
            self._clear_selection(dialog)
            applied = self.apply_answer(dialog, answer.letters, single=single)
        if not applied:
            raise FatalError("答案点击失败", f"目标选项 {answer.letters} 未能选中")
        time.sleep(0.8)

        # 等待平台判分（单选通常立即显示，多选可能需要点“提交”）
        result_text = ""
        correct = None
        feedback_letters: list[str] = []
        deadline = time.monotonic() + 4.0
        submit_clicked = False
        while time.monotonic() < deadline:
            frame = self.screen.grab()
            lines = self.ocr.read(dialog.crop(frame), offset=(dialog.x0, dialog.y0), upscale=1.2)
            text = "\n".join(ln.text for ln in lines)
            letters, correct = parse_feedback(text)
            if letters:
                feedback_letters = letters
                result_text = text
                break
            if "提交" in text and not submit_clicked:
                submit = find_submit_button(lines, dialog)
                if submit is not None:
                    self.mouse.click_rect(submit, "提交答案")
                    submit_clicked = True
                    time.sleep(0.8)
                    continue
            time.sleep(0.6)

        if feedback_letters:
            self.state.record_feedback(question.hash_text(), feedback_letters, question.prompt_text())
            self.log.info(
                "判分结果: %s，正确答案: %s",
                "错误" if correct is False else "正确" if correct is True else "未知",
                feedback_letters,
            )
        else:
            self.log.warning("未读取到判分结果，直接关闭弹题")
        self.state.note_quiz(wrong=(correct is False))

        closed = self.close(frame, dialog)
        if not closed and self._unanswered_open(dialog):
            self._dismiss_unanswered(dialog)
            self._clear_selection(dialog)
            retried = self.apply_answer(dialog, answer.letters, single=single)
            if not retried or self._unanswered_open(dialog):
                raise FatalError(
                    "答案点击失败",
                    f"目标选项 {answer.letters} 仍未选中，已停止以免反复点错关闭按钮",
                )
            closed = self.close(frame, dialog)
        if not closed:
            closed = self._force_close(dialog)
        if not closed:
            if answer.source == "fallback":
                # 随机兜底可能答错；判错后弹窗即使一时关不掉，也不能中断整个挂机
                self.log.error(
                    "随机兜底答错后弹窗仍未关闭，跳过本次弹题并继续恢复播放（请人工复核）"
                )
            else:
                raise FatalError("弹题弹窗无法关闭", "已尝试关闭按钮、右上角 X 和 ESC")
        return {
            "answered": True,
            "closed": closed,
            "correct": correct,
            "letters": answer.letters,
            "feedback": feedback_letters,
            "needs_vision": question.needs_vision,
            "result_text": result_text[:300],
            "fallback": answer.source == "fallback",
        }
