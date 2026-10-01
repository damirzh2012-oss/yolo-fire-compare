"""Инференс YOLO и отрисовка bounding box."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CLASS_COLORS,
    CONFIDENCE_THRESHOLD,
    DEFAULT_BOX_COLOR,
    IOU_MATCH_THRESHOLD,
)

Detection = dict[str, Any]


@lru_cache(maxsize=4)
def load_model(weights_path: str) -> YOLO:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл весов не найден: {path}. "
            "Скачайте yolov8n.pt в models/ или положите туда best.pt."
        )
    return YOLO(str(path))


def run_inference(model: YOLO, frame: np.ndarray) -> list[Detection]:
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections: list[Detection] = []
    if results.boxes is None:
        return detections
    names = results.names
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append(
            {
                "class": names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf[0]),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            }
        )
    return detections


def annotate_frame(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        color = CLASS_COLORS.get(det["class"].lower(), DEFAULT_BOX_COLOR)
        label = f"{det['class']} {det['confidence'] * 100:.0f}%"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, ty), (x1 + tw + 6, ty + th + baseline + 6), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, ty + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
    return out


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def count_false_positives(
    detections: list[Detection],
    ground_truth: list[dict[str, Any]] | None,
    iou_thr: float = IOU_MATCH_THRESHOLD,
) -> int | None:
    if not ground_truth:
        return None
    used: set[int] = set()
    fp = 0
    for det in detections:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(ground_truth):
            if j in used:
                continue
            if det["class"].lower() != str(gt.get("class", "")).lower():
                continue
            score = _iou(det["xyxy"], gt["xyxy"])
            if score > best_iou:
                best_iou = score
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
        else:
            fp += 1
    return fp


def summarize(
    detections: list[Detection],
    inference_time_sec: float,
    ground_truth: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    confs = [d["confidence"] for d in detections]
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    fp = count_false_positives(detections, ground_truth)
    precision = None
    if fp is not None:
        tp = max(0, len(detections) - fp)
        precision = tp / len(detections) if detections else 0.0
    return {
        "detections_count": len(detections),
        "avg_confidence": round(avg_conf, 4),
        "false_positives": fp,
        "precision": precision,
        "inference_time_sec": round(inference_time_sec, 3),
        "detections": detections,
    }


def process_image(
    model: YOLO,
    source_path: Path,
    output_path: Path,
    ground_truth: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise ValueError(f"Не удалось прочитать изображение: {source_path}")
    t0 = time.perf_counter()
    detections = run_inference(model, frame)
    elapsed = time.perf_counter() - t0
    annotated = annotate_frame(frame, detections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return summarize(detections, elapsed, ground_truth)
