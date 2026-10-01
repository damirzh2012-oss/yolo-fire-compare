from roboflow import Roboflow
from ultralytics import YOLO

# 1. Подключаемся к Roboflow и качаем датасет
# Замени API_KEY_НИЖЕ на свой ключ из Roboflow (Settings -> Account -> Private API Key)
rf = Roboflow(api_key="msJWGYQBwoGQEhxzC8T6")

# Указываем твой workspace и project
project = rf.workspace("workspace-v7mdf").project("wildfire-rj82d")
dataset = project.version(11).download("yolov8")

# 2. Загружаем легкую базовую модель YOLOv8
model = YOLO("yolov8n.pt")

# 3. Запускаем обучение на видеокарте
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=5,
    imgsz=320,
    device="cpu"
)

print("Готово! Обучение завершено. Веса сохранены в runs/detect/train/weights/best.pt")
