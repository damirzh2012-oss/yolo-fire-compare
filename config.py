from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

BASE_MODEL_PATH = ROOT / "weights" / "base.pt"
FINETUNED_MODEL_PATH = ROOT / "weights" / "finetuned.pt"

UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
GROUND_TRUTH_DIR = ROOT / "ground_truth"


def resolve_finetuned_path() -> Path:
    if FINETUNED_MODEL_PATH.exists():
        return FINETUNED_MODEL_PATH
    return BASE_MODEL_PATH

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi"}
ALLOWED_EXT = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT

CONFIDENCE_THRESHOLD = 0.15
IOU_MATCH_THRESHOLD = 0.5
VIDEO_FRAME_SKIP = 5
MAX_UPLOAD_MB = 100

CLASS_COLORS = {
    "fire": (0, 140, 255),
    "smoke": (180, 180, 180),
    "flame": (0, 140, 255),
}
DEFAULT_BOX_COLOR = (0, 200, 255)
