import shutil
from pathlib import Path
from ultralytics import YOLO

# 1. Подготовка папок
base_dir = Path("dataset_final")
if base_dir.exists():
    shutil.rmtree(base_dir)

img_dir = base_dir / "images"
lbl_dir = base_dir / "labels"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)

# 2. Собираем скриншоты без дубликатов
kz_dir = Path("dataset_kz")
all_imgs = [
    f
    for f in kz_dir.rglob("*")
    if f.is_file()
    and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
]

seen = set()
unique_imgs = []
for img in all_imgs:
    if img.name not in seen:
        seen.add(img.name)
        unique_imgs.append(img)

for idx, img in enumerate(unique_imgs):
    ext = img.suffix.lower()
    clean_name = f"kz_{idx:03d}{ext}"
    shutil.copy(img, img_dir / clean_name)

print(f"Загружено уникальных скриншотов: {len(unique_imgs)}")

# 3. Авто-разметка скриншотов базовой моделью
model_path = Path("runs/detect/train-8/weights/best.pt")
if not model_path.exists():
    model_path = Path("yolov8n.pt")

model = YOLO(str(model_path))

for img_path in img_dir.glob("*"):
    res = model.predict(img_path, conf=0.15, verbose=False)[0]
    txt_path = lbl_dir / f"{img_path.stem}.txt"

    lines = []
    for box in res.boxes:
        cls = int(box.cls[0])
        x, y, w, h = box.xywhn[0].tolist()
        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    with open(txt_path, "w") as f:
        f.writelines(lines)

print(f"Разметка готова для всех {len(unique_imgs)} файлов!")

# 4. Создание конфигурации
yaml_content = f"""path: {base_dir.resolve().as_posix()}
train: images
val: images

names:
  0: fire
  1: smoke
"""
with open(base_dir / "data.yaml", "w") as f:
    f.write(yaml_content)