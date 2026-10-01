"""Fine-tuning YOLOv8n на датасете лесных пожаров с Roboflow."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

from config import BASE_MODEL_PATH, FINETUNED_MODEL_PATH, MODELS_DIR, ROOT

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "msJWGYQBwoGQEhxzC8T6")
DATASET_DIR = ROOT / "datasets" / "roboflow"
RUNS_DIR = ROOT / "runs" / "detect"
RUN_NAME = "fire_finetune"

# Оставьте пустым — скрипт возьмёт workspace ключа и проект с fire/smoke в названии.
# Либо задайте slug'и явно / через ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION.
DEFAULT_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "")
DEFAULT_PROJECT = os.getenv("ROBOFLOW_PROJECT", "")
DEFAULT_VERSION = int(os.getenv("ROBOFLOW_VERSION", "0"))

FIRE_NAME_RE = re.compile(r"fire|smoke|wildfire|лесн|пожар|дым", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Дообучение YOLOv8n на датасете пожаров (Roboflow)")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION, help="0 = последняя версия")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _project_slug(project_id: str) -> str:
    return project_id.split("/")[-1]


def _pick_project_slug(projects: list[dict], explicit: str) -> str:
    if explicit:
        return explicit
    if not projects:
        raise RuntimeError(
            "В workspace нет проектов. Укажите --workspace и --project "
            "(slug из Roboflow Universe / вашего аккаунта)."
        )
    scored = []
    for item in projects:
        slug = _project_slug(str(item.get("id") or item.get("name") or ""))
        name = str(item.get("name") or slug)
        score = 2 if FIRE_NAME_RE.search(name) or FIRE_NAME_RE.search(slug) else 0
        scored.append((score, slug, name))
    scored.sort(key=lambda row: (-row[0], row[1]))
    best_score, slug, name = scored[0]
    if best_score == 0:
        print(f"Проект про пожары не найден, берём первый: {name} ({slug})")
    else:
        print(f"Выбран проект: {name} ({slug})")
    return slug


def _latest_version_number(project) -> int:
    getter = getattr(project, "versions", None)
    versions = getter() if callable(getter) else (getter or [])
    numbers = []
    for ver in versions or []:
        raw = getattr(ver, "version", None) or getattr(ver, "id", None) or ver
        if isinstance(raw, dict):
            raw = raw.get("id") or raw.get("version") or raw.get("name")
        try:
            numbers.append(int(str(raw).split("/")[-1]))
        except (TypeError, ValueError):
            continue
    return max(numbers) if numbers else 1


def download_roboflow_dataset(workspace: str, project: str, version: int) -> Path:
    from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    print(f"Roboflow workspace: {getattr(ws, 'name', None) or workspace or '(default)'}")

    projects = ws.list_projects() if hasattr(ws, "list_projects") else ws.projects()
    slug = _pick_project_slug(projects, project)
    rf_project = ws.project(slug)

    ver_num = version if version > 0 else _latest_version_number(rf_project)
    print(f"Скачивание версии {ver_num} в формате yolov8 → {DATASET_DIR}")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    dataset = rf_project.version(ver_num).download("yolov8", location=str(DATASET_DIR))
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        matches = list(Path(dataset.location).rglob("data.yaml"))
        if not matches:
            raise FileNotFoundError("После загрузки не найден data.yaml")
        data_yaml = matches[0]
    print(f"data.yaml: {data_yaml}")
    return data_yaml


def resolve_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch не установлен") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна. Для RTX 5070 Ti нужна сборка PyTorch с поддержкой GPU "
            "и актуальный драйвер NVIDIA. Проверьте: python -c \"import torch; print(torch.cuda.is_available())\""
        )
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} ({torch.version.cuda=})")
    return "cuda"


def export_best_weights(train_dir: Path) -> Path:
    src = train_dir / "weights" / "best.pt"
    if not src.exists():
        raise FileNotFoundError(f"Не найден {src}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, FINETUNED_MODEL_PATH)
    print(f"Скопировано: {src} → {FINETUNED_MODEL_PATH}")
    return FINETUNED_MODEL_PATH


def main() -> int:
    args = parse_args()
    if not BASE_MODEL_PATH.exists():
        print(f"Нет базовых весов {BASE_MODEL_PATH}. Сначала скачайте yolov8n.pt в models/.", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    data_yaml = download_roboflow_dataset(args.workspace, args.project, args.version)

    model = YOLO(str(BASE_MODEL_PATH))
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        plots=True,
    )
    train_dir = Path(results.save_dir)
    export_best_weights(train_dir)
    print("Обучение завершено. Веб-приложение подхватит models/best.pt как доработанную модель.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
