import os
from pathlib import Path
from ultralytics import YOLO

# Путь к обученной модели (train-8) и к папке с 40 фото
MODEL_PATH = "runs/detect/train-8/weights/best.pt"
IMAGES_DIR = Path("dataset_kz")

model = YOLO(MODEL_PATH)

# Размечаем все картинки в папке dataset_kz
for img_path in IMAGES_DIR.glob("*.*"):
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        results = model.predict(source=str(img_path), conf=0.25)
        
        # Сохраняем лейблы в формате YOLO (.txt)
        txt_path = img_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    # Нормализованные координаты xywh
                    x, y, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print("Разметка 40 фото завершена!")