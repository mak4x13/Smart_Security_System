import os

BASE_DB = "backend/db"
PERSONS_CSV = os.path.join(BASE_DB, "persons.csv")
ATTENDANCE_CSV = os.path.join(BASE_DB, "attendance.csv")
CHROMA_PATH = os.path.join(BASE_DB, "chromadb_data")

THRESHOLD = 0.6
SAMPLES_REQUIRED = 10
