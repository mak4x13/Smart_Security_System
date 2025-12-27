import csv
import os
from datetime import datetime, date
from backend.config import ATTENDANCE_CSV

# In-memory guard to avoid duplicates per day
SEEN_TODAY = set()

def log_attendance(person_id):
    today = date.today().isoformat()
    key = (person_id, today)

    if key in SEEN_TODAY:
        return  # already logged today

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
