# YOLO Fire Compare

MVP для сравнения **базовой** и **дообученной** YOLO-моделей на задаче обнаружения дыма и огня (фото/видео лесных и степных пожаров).

## Стек

- Backend: FastAPI
- Детекция: Ultralytics YOLOv8
- Видео: OpenCV
- Frontend: HTML / CSS / Vanilla JS, Chart.js

## Запуск

```bash
cd yolo-fire-compare
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Linux / macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Откройте в браузере: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Первый запуск скачает веса `yolov8n.pt` (нужен интернет). Положите файл в `models/yolov8n.pt` или оставьте автозагрузку Ultralytics.

## Модели (заглушки)

В `config.py` обе модели пока указывают на `yolov8n.pt`, чтобы можно было проверить интерфейс до обучения custom-весов:

```python
BASE_MODEL_PATH = ROOT / "models" / "yolov8n.pt"
FINETUNED_MODEL_PATH = ROOT / "models" / "yolov8n.pt"
```

После дообучения замените `FINETUNED_MODEL_PATH` на путь к `custom_model.pt`.

Предобученный COCO `yolov8n` **не содержит** классов smoke/fire. На тестовых кадрах пожара боксы могут быть пустыми или нерелевантными — это ожидаемо до появления дообученной модели.

## Ground truth (опционально)

Если рядом с загруженным файлом есть JSON с эталоном, в статистике появятся ложные срабатывания и прирост точности по precision.

Положите файл в `ground_truth/<stem>.json` (stem — имя загруженного файла без расширения) или в `uploads/` с тем же stem:

```json
{
  "boxes": [
    { "class": "fire", "xyxy": [100, 80, 400, 320] },
    { "class": "smoke", "xyxy": [40, 20, 220, 180] }
  ]
}
```

Сопоставление: IoU ≥ 0.5 и совпадение имени класса. Без GT ложные срабатывания = «нет GT», а «Прирост точности» считается как относительное изменение среднего confidence.

## Видео

Кадры обрабатываются каждый N-й (`VIDEO_FRAME_SKIP` в `config.py`, по умолчанию 5). Пропущенные кадры получают последние боксы, затем поток собирается через `cv2.VideoWriter`.

## API

- `POST /upload` — файл → `uploads/`, JSON с `filename` и `url`
- `POST /predict` — `{ "filename": "..." }` → два файла в `outputs/` и статистика

## Структура

```
main.py            # FastAPI
config.py          # пути к весам и пороги
detection.py       # инференс и отрисовка боксов
video_utils.py     # покадровая обработка видео
templates/index.html
static/style.css
static/app.js
uploads/  outputs/  models/  ground_truth/
```
