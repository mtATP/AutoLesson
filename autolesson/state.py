from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from .types import Lesson, normalize_title

log = logging.getLogger("autolesson.state")

STATE_VERSION = 1


def _now() -> float:
    return time.time()


class State:
    """运行状态与答题缓存，原子落盘，进程中断也不会损坏。"""

    def __init__(self, path: Path, logger: Optional[logging.Logger] = None):
        self.path = Path(path)
        self.log = logger or log
        self.data: dict[str, Any] = {
            "version": STATE_VERSION,
            "lessons": {},
            "practices": {},
            "answers": {},
            "stats": {
                "lessons_completed": 0,
                "quizzes": 0,
                "quizzes_wrong": 0,
                "llm_calls": 0,
                "llm_tokens": 0,
                "watch_seconds": 0.0,
                "runs": 0,
            },
        }
        self.load()

    # ------------------------------------------------------------------ IO
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("根节点不是对象")
            self.data.update(data)
            if not isinstance(self.data.get("lessons"), dict):
                self.data["lessons"] = {}
            if not isinstance(self.data.get("answers"), dict):
                self.data["answers"] = {}
            self.data.setdefault("lessons", {})
            self.data.setdefault("practices", {})
            self.data.setdefault("answers", {})
            if not isinstance(self.data.get("stats"), dict):
                self.data["stats"] = {}
            stats = self.data["stats"]
            for key, value in self._default_stats().items():
                stats.setdefault(key, value)
            self.data["version"] = STATE_VERSION
            self.log.info("已加载状态: %s (已完成 %d 节)", self.path, len(self.data["lessons"]))
        except Exception as exc:  # noqa: BLE001
            backup = self.path.with_suffix(self.path.suffix + f".corrupt-{int(time.time())}")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            self.log.warning("状态文件损坏，已备份到 %s: %s", backup, exc)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=1)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("保存状态失败: %s", exc)

    @staticmethod
    def _default_stats() -> dict[str, Any]:
        return {
            "lessons_completed": 0,
            "quizzes": 0,
            "quizzes_wrong": 0,
            "llm_calls": 0,
            "llm_tokens": 0,
            "watch_seconds": 0.0,
            "runs": 0,
        }

    # ------------------------------------------------------------------ 课程
    @property
    def lessons(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("lessons", {})

    def is_completed(self, lesson: Lesson | str) -> bool:
        key = lesson.key if isinstance(lesson, Lesson) else str(lesson)
        if key in self.lessons:
            return True
        title = normalize_title(lesson.title) if isinstance(lesson, Lesson) else ""
        if title:
            for meta in self.lessons.values():
                if isinstance(meta, dict) and normalize_title(str(meta.get("title", ""))) == title:
                    return True
        return False

    def mark_completed(self, lesson: Lesson) -> None:
        key = lesson.key
        existed = key in self.lessons
        self.lessons[key] = {
            "index": lesson.index,
            "title": lesson.title,
            "duration": lesson.duration,
            "completed_at": _now(),
        }
        if not existed:
            self.data["stats"]["lessons_completed"] = int(self.data["stats"].get("lessons_completed", 0)) + 1
        self.save()

    def completed_count(self) -> int:
        return len(self.lessons)

    # ------------------------------------------------------------------ 平时测试记录
    @property
    def practices(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("practices", {})

    def is_practice_done(self, key: str) -> bool:
        return str(key) in self.practices

    def mark_practice(self, key: str, chapter: str = "", title: str = "") -> None:
        self.practices[str(key)] = {
            "chapter": chapter,
            "title": title,
            "saved_at": _now(),
        }
        self.save()

    def clear_records(self) -> None:
        """只清空已完成记录（视频/练习），保留答题缓存。"""
        self.data["lessons"] = {}
        self.data["practices"] = {}
        self.save()
        self.log.info("已清空完成记录（视频/练习），答题缓存保留")

    # ------------------------------------------------------------------ 答题缓存
    @property
    def answers(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("answers", {})

    def cached_answer(self, qhash: str) -> Optional[dict[str, Any]]:
        value = self.answers.get(qhash)
        return value if isinstance(value, dict) else None

    def cache_answer(self, qhash: str, letters: list[str], confidence: float, question: str) -> None:
        self.answers[qhash] = {
            "letters": list(letters),
            "confidence": float(confidence),
            "question": question[:500],
            "updated_at": _now(),
        }
        if len(self.answers) > 1000:
            oldest = sorted(self.answers.items(), key=lambda kv: kv[1].get("updated_at", 0))[:200]
            for key, _value in oldest:
                self.answers.pop(key, None)
        self.save()

    def record_feedback(self, qhash: str, letters: list[str], question: str) -> None:
        """平台公布正确答案后写入缓存，下次遇到同题不再调用大模型。"""
        if not letters:
            return
        self.cache_answer(qhash, letters, 1.0, question)

    # ------------------------------------------------------------------ 统计
    def note_quiz(self, wrong: bool = False) -> None:
        stats = self.data["stats"]
        stats["quizzes"] = int(stats.get("quizzes", 0)) + 1
        if wrong:
            stats["quizzes_wrong"] = int(stats.get("quizzes_wrong", 0)) + 1
        self.save()

    def note_llm(self, usage: Optional[dict[str, Any]] = None) -> None:
        stats = self.data["stats"]
        stats["llm_calls"] = int(stats.get("llm_calls", 0)) + 1
        if usage:
            total = usage.get("total_tokens")
            if isinstance(total, (int, float)):
                stats["llm_tokens"] = int(stats.get("llm_tokens", 0)) + int(total)
        self.save()

    def note_watch(self, seconds: float) -> None:
        stats = self.data["stats"]
        stats["watch_seconds"] = float(stats.get("watch_seconds", 0.0)) + max(0.0, seconds)
        self.save()

    def note_run(self) -> None:
        self.data["stats"]["runs"] = int(self.data["stats"].get("runs", 0)) + 1
        self.save()

    def summary(self) -> str:
        stats = self.data["stats"]
        minutes = float(stats.get("watch_seconds", 0.0)) / 60.0
        return (
            f"累计: 完成 {stats.get('lessons_completed', 0)} 节, "
            f"答题 {stats.get('quizzes', 0)} 次(错 {stats.get('quizzes_wrong', 0)}), "
            f"大模型调用 {stats.get('llm_calls', 0)} 次/{stats.get('llm_tokens', 0)} tokens, "
            f"观看 {minutes:.1f} 分钟"
        )
