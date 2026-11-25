"""Background work (analysis, previews, export) with pollable progress."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import uuid
from typing import Callable


class Job:
    def __init__(self, kind: str, target: str | None = None):
        self.id = f"job_{uuid.uuid4().hex[:8]}"
        self.kind = kind
        self.target = target
        self.status = "running"
        self.progress = 0.0
        self.message = "Starting"
        self.result: dict | None = None
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None

    def update(self, progress: float, message: str | None = None) -> None:
        self.progress = max(0.0, min(1.0, progress))
        if message:
            self.message = message

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "target": self.target, "status": self.status,
            "progress": round(self.progress, 4), "message": self.message,
            "result": self.result, "error": self.error,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
        }


class JobRunner:
    def __init__(self, max_jobs: int = 50):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.max_jobs = max_jobs

    def start(self, kind: str, fn: Callable[[Job], dict | None], target: str | None = None) -> Job:
        job = Job(kind, target)
        with self.lock:
            self.jobs[job.id] = job
            done = [j for j in self.jobs.values() if j.status != "running"]
            for old in sorted(done, key=lambda j: j.started)[: max(0, len(self.jobs) - self.max_jobs)]:
                del self.jobs[old.id]

        def run() -> None:
            try:
                job.result = fn(job) or {}
                job.status = "done"
                job.progress = 1.0
            except Exception as exc:  # reported to the UI
                job.status = "error"
                job.error = str(exc) or exc.__class__.__name__
                print(f"[{job.kind}] {job.error}", file=sys.stderr)
                if os.environ.get("WTG_DEBUG"):
                    traceback.print_exc()
            finally:
                job.finished = time.time()

        threading.Thread(target=run, daemon=True, name=job.id).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def active(self) -> list[dict]:
        return [j.to_dict() for j in self.jobs.values()
                if j.status == "running" or (j.finished and time.time() - j.finished < 30)]
