"""Bounded lossless image writing without dropping or reordering capture frames."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np


class ImageWriteError(RuntimeError):
    """At least one captured image could not be written."""


def _write_png_frame(images: tuple[tuple[Path, np.ndarray], ...]) -> float:
    from PIL import Image

    started = perf_counter()
    for path, rgb in images:
        Image.fromarray(rgb).save(path, compress_level=1)
    return perf_counter() - started


class BoundedImageWriter:
    """A single producer owns frame order; workers only write disjoint PNG paths.

    The bound includes running jobs. Backpressure waits before copying another
    frame, so retained RGB arrays never exceed ``max_pending_frames`` bundles.
    A successful close means every submitted PNG has finished writing.
    """

    def __init__(self, workers: int, max_pending_frames: int):
        if type(workers) is not int or workers < 1:
            raise ValueError("Image workers must be a positive integer")
        if type(max_pending_frames) is not int or max_pending_frames < 1:
            raise ValueError("Image writer queue must be a positive integer")
        self.workers = workers
        self.max_pending_frames = max_pending_frames
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nepes-png")
        self._pending: deque[Future] = deque()
        self._failure: Exception | None = None
        self._closed = False
        self._submitted_frames = 0
        self._max_pending_observed = 0
        self._backpressure_seconds = 0.
        self._worker_seconds = 0.

    @property
    def metrics(self) -> dict:
        return {"workers": self.workers, "max_pending_frames": self.max_pending_frames,
                "submitted_frames": self._submitted_frames,
                "max_pending_frames_observed": self._max_pending_observed,
                "backpressure_seconds": self._backpressure_seconds,
                "worker_seconds": self._worker_seconds, "closed": self._closed,
                "failed": self._failure is not None}

    def _collect(self, future: Future) -> None:
        try:
            self._worker_seconds += future.result()
        except Exception as exc:
            if self._failure is None:
                self._failure = exc

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise ImageWriteError(f"Captured PNG write failed: {self._failure}") from self._failure

    def submit(self, images: Mapping[Path, np.ndarray]) -> None:
        if self._closed:
            raise RuntimeError("Image writer is closed")
        # Collect all completed work, including failures finishing out of order.
        remaining = deque()
        for future in self._pending:
            if future.done():
                self._collect(future)
            else:
                remaining.append(future)
        self._pending = remaining
        self._raise_failure()
        if len(self._pending) >= self.max_pending_frames:
            started = perf_counter()
            self._collect(self._pending.popleft())
            self._backpressure_seconds += perf_counter() - started
            self._raise_failure()
        # Never retain caller-owned buffers: the next camera frame may reuse
        # storage immediately after submit returns.
        owned = tuple((Path(path), np.array(rgb, copy=True, order="C"))
                      for path, rgb in images.items())
        self._pending.append(self._executor.submit(_write_png_frame, owned))
        self._submitted_frames += 1
        self._max_pending_observed = max(self._max_pending_observed, len(self._pending))

    def close(self) -> None:
        if not self._closed:
            # Join ALL jobs even when an earlier future failed. Cleanup or a
            # staging-directory rename cannot race a still-running PNG writer.
            try:
                while self._pending:
                    self._collect(self._pending.popleft())
            finally:
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._closed = True
        self._raise_failure()
