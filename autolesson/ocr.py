from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import cv2
import numpy as np

from .types import OcrLine, Rect

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")


class OcrUnavailable(RuntimeError):
    pass


class OcrEngine:
    """RapidOCR 封装，兼容 rapidocr 3.x 与旧版 rapidocr_onnxruntime。"""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        min_score: float = 0.45,
        intra_threads: int = 1,
        inter_threads: int = 1,
        cv_threads: int = 1,
        det_limit_side: int = 416,
        use_cls: bool = False,
    ):
        self.log = logger or logging.getLogger("autolesson.ocr")
        self.min_score = min_score
        self.intra_threads = max(1, int(intra_threads))
        self.inter_threads = max(1, int(inter_threads))
        self.cv_threads = max(1, int(cv_threads))
        self.det_limit_side = max(160, int(det_limit_side))
        self.use_cls = bool(use_cls)
        self._engine: Any = None
        self._api = ""
        self._init_error: Optional[str] = None
        # 截图/连通域/缩放都是小任务，单线程即可；避免 OpenCV 自己开满所有核心
        try:
            cv2.setNumThreads(self.cv_threads)
            cv2.ocl.setUseOpenCL(False)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ 初始化
    def _ensure(self) -> None:
        if self._engine is not None:
            return
        if self._init_error:
            raise OcrUnavailable(self._init_error)
        try:
            import onnxruntime as ort  # type: ignore

            ort.set_default_logger_severity(3)
        except Exception:  # noqa: BLE001
            pass
        self._silence_rapidocr()
        try:
            from rapidocr import RapidOCR  # type: ignore
        except Exception as first_exc:  # noqa: BLE001
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore

                try:
                    self._engine = RapidOCR(
                        intra_op_num_threads=self.intra_threads,
                        use_cls=self.use_cls,
                    )
                except TypeError:
                    self._engine = RapidOCR()
                self._api = "old"
                self._silence_rapidocr()
                self.log.info("OCR 引擎: rapidocr_onnxruntime")
                return
            except Exception as second_exc:  # noqa: BLE001
                self._init_error = (
                    "无法加载 RapidOCR。请执行: pip install rapidocr onnxruntime\n"
                    f"详细错误: {first_exc} / {second_exc}"
                )
                raise OcrUnavailable(self._init_error) from second_exc

        params = {
            "EngineConfig.onnxruntime.intra_op_num_threads": self.intra_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": self.inter_threads,
            "Det.limit_side_len": self.det_limit_side,
            "Global.use_cls": self.use_cls,
            "Global.log_level": "error",
        }
        try:
            self._engine = RapidOCR(params=params)
        except TypeError:
            # 极旧版本不接受 params，退回默认构造（线程数无法限制）
            self.log.warning("当前 RapidOCR 不支持线程参数，将使用默认线程数")
            self._engine = RapidOCR()
        self._api = "new"
        self._silence_rapidocr()
        self.log.info(
            "OCR 引擎: rapidocr (onnxruntime) threads=%d/%d cv=%d det_side=%d use_cls=%s",
            self.intra_threads,
            self.inter_threads,
            self.cv_threads,
            self.det_limit_side,
            self.use_cls,
        )

    @staticmethod
    def _silence_rapidocr() -> None:
        for name in ("RapidOCR", "rapidocr", "RapidOCR.utils", "rapidocr.utils"):
            logger = logging.getLogger(name)
            logger.setLevel(logging.ERROR)
            for handler in logger.handlers:
                handler.setLevel(logging.ERROR)

    def warmup(self) -> None:
        try:
            self.read(np.full((40, 120, 3), 255, dtype=np.uint8), min_score=0.9)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("OCR 预热失败: %s", exc)

    @property
    def api(self) -> str:
        return self._api

    # ------------------------------------------------------------------ 调用
    def read(
        self,
        img: np.ndarray,
        offset: tuple[int, int] = (0, 0),
        upscale: Optional[float] = None,
        min_score: Optional[float] = None,
    ) -> list[OcrLine]:
        if img is None or img.size == 0:
            return []
        self._ensure()
        prepared, scale = self._prepare(img, upscale)
        t0 = time.monotonic()
        try:
            out = self._engine(prepared, use_cls=self.use_cls)
        except TypeError:
            # 旧版签名差异兜底
            out = self._engine(prepared)
        elapsed = time.monotonic() - t0
        rows, txts, scores = self._unpack(out)
        threshold = self.min_score if min_score is None else min_score

        lines: list[OcrLine] = []
        if rows is None:
            return []
        if txts is not None:
            for box, text, score in zip(rows, txts, scores):
                text = str(text).strip()
                if not text:
                    continue
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    score_f = 1.0
                if score_f < threshold:
                    continue
                rect = _box_to_rect(box, scale, offset)
                if rect.width <= 0 or rect.height <= 0:
                    continue
                lines.append(OcrLine(text=text, rect=rect, score=score_f))
        else:
            for item in rows:
                if not item:
                    continue
                try:
                    box, text, score = item[0], item[1], item[2]
                except (IndexError, TypeError):
                    continue
                text = str(text).strip()
                if not text or float(score) < threshold:
                    continue
                rect = _box_to_rect(box, scale, offset)
                if rect.width > 0 and rect.height > 0:
                    lines.append(OcrLine(text=text, rect=rect, score=float(score)))

        self.log.debug("OCR %.2fs 缩放%.2f %dx%d -> %d 行", elapsed, scale, img.shape[1], img.shape[0], len(lines))
        return lines

    def read_text(self, img: np.ndarray, **kwargs: Any) -> str:
        return "\n".join(line.text for line in self.read(img, **kwargs))

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _prepare(img: np.ndarray, upscale: Optional[float]) -> tuple[np.ndarray, float]:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        scale = float(upscale or 1.0)
        if scale > 1.01:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(img), scale

    def _unpack(self, out: Any) -> tuple[Any, Optional[Any], Optional[Any]]:
        # rapidocr 3.x: RapidOCROutput
        if hasattr(out, "boxes") and hasattr(out, "txts"):
            boxes = out.boxes if out.boxes is not None else []
            txts = out.txts
            scores = out.scores
            if txts is None:
                return boxes, None, None
            return boxes, txts, scores
        # rapidocr_onnxruntime: (result, elapse)
        if isinstance(out, tuple) and len(out) == 2:
            out = out[0]
        if out is None:
            return [], None, None
        if isinstance(out, dict):
            return out.get("boxes", []), out.get("txts"), out.get("scores")
        return out, None, None


def _box_to_rect(box: Any, scale: float, offset: tuple[int, int]) -> Rect:
    pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return Rect(0, 0, 0, 0)
    xs = pts[:, 0] / scale + offset[0]
    ys = pts[:, 1] / scale + offset[1]
    return Rect(int(round(xs.min())), int(round(ys.min())), int(round(xs.max())), int(round(ys.max())))


def sort_lines(lines: list[OcrLine], row_tolerance: int = 8) -> list[OcrLine]:
    """按阅读顺序排序：先按行聚类，再从左到右。"""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda ln: (ln.cy, ln.cx))
    rows: list[list[OcrLine]] = []
    for line in ordered:
        if rows and abs(line.cy - rows[-1][0].cy) <= row_tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
    out: list[OcrLine] = []
    for row in rows:
        out.extend(sorted(row, key=lambda ln: ln.cx))
    return out
