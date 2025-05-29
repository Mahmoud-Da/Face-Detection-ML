import torch
from pathlib import Path

# ---- Device ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- MTCNN Model Parameters ----
# Default for facenet-pytorch, but detection can work on various sizes
MTCNN_IMAGE_SIZE = 160
MTCNN_MARGIN = 0
MTCNN_MIN_FACE_SIZE = 20
MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]  # P-Net, R-Net, O-Net thresholds
MTCNN_FACTOR = 0.709  # Scale factor
MTCNN_KEEP_ALL = True  # Detect all faces, or only the largest/most probable
MTCNN_POST_PROCESS = True  # Apply post-processing to detection

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
INPUT_IMAGE_DIR = BASE_DIR / "input"
OUTPUT_IMAGE_DIR = BASE_DIR / "output"

# Ensure output directory exists
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---- Inference Parameters ----
# Example: Default image to process if none is provided via command line
# Change this to an image in images_input/
DEFAULT_IMAGE_NAME = "people_at_coffee_shop.jpg"
# Only display/draw boxes with confidence above this
CONFIDENCE_THRESHOLD_DISPLAY = 0.9
