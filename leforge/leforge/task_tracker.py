"""Global task tracker for long-running operations.

Tracks ongoing tasks (SSH install, recording, training) so the frontend
can reconnect after page navigation or refresh.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

TaskType = Literal["ssh_install", "recording", "training"]
TaskStatus = Literal["running", "done", "failed"]


@dataclass
class TaskInfo:
    task_id: str
    task_type: TaskType
    status: TaskStatus = "running"
    label: str = ""
    message: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "label": self.label,
            "message": self.message,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class TaskTracker:
    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def create(
        self,
        task_id: str,
        task_type: TaskType,
        label: str = "",
        metadata: dict | None = None,
    ) -> TaskInfo:
        with self._lock:
            task = TaskInfo(
                task_id=task_id,
                task_type=task_type,
                label=label,
                metadata=metadata or {},
            )
            self._tasks[task_id] = task
            return task

    def get(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update(
        self,
        task_id: str,
        status: TaskStatus | None = None,
        label: str | None = None,
        message: str | None = None,
        metadata: dict | None = None,
    ) -> TaskInfo | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if status is not None:
                task.status = status
            if label is not None:
                task.label = label
            if message is not None:
                task.message = message
            if metadata is not None:
                task.metadata.update(metadata)
            task.updated_at = time.time()
            return task

    def remove(self, task_id: str):
        with self._lock:
            self._tasks.pop(task_id, None)

    def list_active(self) -> list[dict]:
        with self._lock:
            return [
                t.to_dict()
                for t in self._tasks.values()
                if t.status == "running"
            ]

    def list_all(self) -> list[dict]:
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]


task_tracker = TaskTracker()
