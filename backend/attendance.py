import csv
import os
from datetime import datetime, date
from backend.config import ATTENDANCE_CSV

# In-memory guard to avoid duplicates per day (per process)
SEEN_TODAY = set()
SEEN_DAY = date.today().isoformat()

def _already_logged_today(person_id: str, today: str) -> bool:
    if not os.path.exists(ATTENDANCE_CSV):
        return False
    with open(ATTENDANCE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("person_id") == person_id and row.get("timestamp", "").startswith(today):
                return True
    return False

def log_attendance(person_id):
    global SEEN_DAY
    today = date.today().isoformat()
    if SEEN_DAY != today:
        SEEN_TODAY.clear()
        SEEN_DAY = today
    key = (person_id, today)

    if key in SEEN_TODAY:
        return  # already logged today
    if _already_logged_today(person_id, today):
        SEEN_TODAY.add(key)
        return

    os.makedirs(os.path.dirname(ATTENDANCE_CSV), exist_ok=True)

    file_exists = os.path.exists(ATTENDANCE_CSV)
    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "person_id", "status", "source"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            person_id,
            "present",
            "webcam"
        ])

    SEEN_TODAY.add(key)


def read_attendance(today_only=True):
    if not os.path.exists(ATTENDANCE_CSV):
        return []

    today = date.today().isoformat()
    records = []

    with open(ATTENDANCE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if today_only:
                if row["timestamp"].startswith(today):
                    records.append(row)
            else:
                records.append(row)

    return records
