from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    ALLOWED_EXT,
    ALLOWED_IMAGE_EXT,
    ALLOWED_VIDEO_EXT,
    BASE_MODEL_PATH,
    GROUND_TRUTH_DIR,
    MAX_UPLOAD_MB,
    OUTPUT_DIR,
    ROOT,
    UPLOAD_DIR,
    VIDEO_FRAME_SKIP,
    resolve_finetuned_path,
)
from detection import load_model, process_image, run_inference
from video_utils import process_video

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "static").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="YOLO Fire Compare", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


class PredictRequest(BaseModel):
    filename: str


def _media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in ALLOWED_IMAGE_EXT:
        return "image"
    if ext in ALLOWED_VIDEO_EXT:
        return "video"
    raise HTTPException(400, "Неподдерживаемый формат")


def _safe_upload_path(filename: str) -> Path:
    name = Path(filename).name
    path = (UPLOAD_DIR / name).resolve()
    if path.parent != UPLOAD_DIR.resolve():
        raise HTTPException(400, "Некорректный путь")
    if not path.exists():
        raise HTTPException(404, "Файл не найден. Сначала загрузите его через /upload")
    return path


def _load_ground_truth(stem: str) -> list[dict] | None:
    candidates = [
        GROUND_TRUTH_DIR / f"{stem}.json",
        UPLOAD_DIR / f"{stem}.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            boxes = data.get("boxes", data if isinstance(data, list) else [])
            return boxes or None
    return None


def _accuracy_gain(base: dict, finetuned: dict) -> float:
    if base.get("precision") is not None and finetuned.get("precision") is not None:
        return round((finetuned["precision"] - base["precision"]) * 100, 2)
    b = base.get("avg_confidence") or 0.0
    f = finetuned.get("avg_confidence") or 0.0
    if b == 0:
        return 0.0 if f == 0 else 100.0
    return round(((f - b) / b) * 100, 2)


@app.get("/")
async def index():
    return FileResponse(ROOT / "templates" / "index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Файл не выбран")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "Поддерживаются только jpg, png, mp4, avi")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {MAX_UPLOAD_MB} МБ")

    stored_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / stored_name
    dest.write_bytes(data)
    return {
        "filename": stored_name,
        "original_name": file.filename,
        "media_type": _media_type(dest),
        "url": f"/uploads/{stored_name}",
    }


@app.post("/predict")
async def predict(body: PredictRequest):
    src = _safe_upload_path(body.filename)
    media = _media_type(src)
    gt = _load_ground_truth(src.stem)
    ext = src.suffix.lower() if media == "image" else ".mp4"
    ft_path = resolve_finetuned_path()

    if not BASE_MODEL_PATH.exists():
        raise HTTPException(
            500,
            f"Нет файла базовой модели: {BASE_MODEL_PATH}",
        )

    try:
        base_model = load_model(str(BASE_MODEL_PATH))
        ft_model = load_model(str(ft_path))

        base_out = OUTPUT_DIR / f"{src.stem}_base{ext}"
        ft_out = OUTPUT_DIR / f"{src.stem}_finetuned{ext}"

        if media == "image":
            base_stats = process_image(base_model, src, base_out, gt)
            ft_stats = process_image(ft_model, src, ft_out, gt)
        else:
            base_stats = process_video(
                src,
                base_out,
                lambda frame, m=base_model: run_inference(m, frame),
                VIDEO_FRAME_SKIP,
                gt,
            )
            ft_stats = process_video(
                src,
                ft_out,
                lambda frame, m=ft_model: run_inference(m, frame),
                VIDEO_FRAME_SKIP,
                gt,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Ошибка обработки: {exc}") from exc

    for stats in (base_stats, ft_stats):
        stats.pop("detections", None)

    return {
        "media_type": media,
        "has_ground_truth": gt is not None,
        "base": {
            **base_stats,
            "output_url": f"/outputs/{base_out.name}",
            "model": BASE_MODEL_PATH.name,
        },
        "finetuned": {
            **ft_stats,
            "output_url": f"/outputs/{ft_out.name}",
            "model": ft_path.name,
            "using_stub": ft_path == BASE_MODEL_PATH,
        },
        "accuracy_gain_pct": _accuracy_gain(base_stats, ft_stats),
    }
