"""泛雅平台的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from autolesson.types import Rect

STATUS_COMPLETED = "completed"
STATUS_PENDING = "pending"
STATUS_UNKNOWN = "unknown"


@dataclass
class FanyaItem:
    """目录里的一个知识/任务点（通常是三级叶子节点）。"""

    index: str
    title: str
    level: int
    status: str
    row: Rect
    title_rect: Rect
    current: bool = False
    pending_count: Optional[int] = None

    @property
    def key(self) -> str:
        return f"{self.index}|{self.title}"

    @property
    def completed(self) -> bool:
        return self.status == STATUS_COMPLETED

    @property
    def pending(self) -> bool:
        return self.status == STATUS_PENDING

    def __str__(self) -> str:  # pragma: no cover - 日志辅助
        return f"{self.index} {self.title} [{self.status}]"
