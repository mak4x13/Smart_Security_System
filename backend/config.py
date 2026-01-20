import os

BASE_DB = "backend/db"
PERSONS_CSV = os.path.join(BASE_DB, "persons.csv")
ATTENDANCE_CSV = os.path.join(BASE_DB, "attendance.csv")
CHROMA_PATH = os.path.join(BASE_DB, "chromadb_data")

THRESHOLD = 0.6
SAMPLES_REQUIRED = 10

# Detection tuning (safe defaults for CPU webcams).
DETECTION_MODEL_SELECTION = 1  # 0=short range, 1=long range
DETECTION_MIN_CONFIDENCE = 0.5
DETECTION_ENHANCE = True
DETECTION_UPSCALE = 1.2
DETECTION_FRAME_SKIP = 3
DETECTION_HOLD_FRAMES = 2
