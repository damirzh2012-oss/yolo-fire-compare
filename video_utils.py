"""Покадровая обработка видео через OpenCV."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from detection import Detection, annotate_frame, summarize


InferFn = Callable[[np.ndarray], list[Detection]]


def process_video(
    source_path: Path,
    output_path: Path,
    infer_fn: InferFn,
    frame_skip: int = 5,
    ground_truth: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = _open_writer(output_path, fps, width, height)
    last_dets: list[Detection] = []
    collected: list[Detection] = []
    infer_time = 0.0
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, frame_skip) == 0:
                t0 = time.perf_counter()
                last_dets = infer_fn(frame)
                infer_time += time.perf_counter() - t0
                collected.extend(last_dets)
            writer.write(annotate_frame(frame, last_dets))
            idx += 1
    finally:
        cap.release()
        writer.release()

    if idx == 0:
        raise ValueError("Видео не содержит кадров")

    return summarize(collected, infer_time, ground_truth)


def _open_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for fourcc_name in ("avc1", "H264", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Не удалось создать VideoWriter (проверьте кодеки OpenCV)")
