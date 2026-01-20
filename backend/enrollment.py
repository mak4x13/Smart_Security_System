import uuid
import csv
import os
from datetime import datetime
import numpy as np
import chromadb
from backend.config import PERSONS_CSV, CHROMA_PATH, SAMPLES_REQUIRED
from backend.recognition import recognize_face


client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("face_embeddings")

# Enrollment session state
CURRENT_SESSION = {
    "person_id": None,
    "embeddings": [],
    "count": 0
}

def _ensure_enrolled_at_header():
    if not os.path.exists(PERSONS_CSV):
        return
    with open(PERSONS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = rows[0]
    if "enrolled_at" in header:
        return
    header.append("enrolled_at")
    for row in rows[1:]:
        row.append("")
    with open(PERSONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def generate_person_id():
    pid = f"PID_{uuid.uuid4().hex[:8]}"
    CURRENT_SESSION.update({
        "person_id": pid,
        "embeddings": [],
        "count": 0
    })
    return pid

def add_embedding(embedding):
    CURRENT_SESSION["embeddings"].append(embedding.tolist())
    CURRENT_SESSION["count"] += 1

    done = CURRENT_SESSION["count"] >= SAMPLES_REQUIRED
    return done, CURRENT_SESSION["count"]


def finalize_enrollment(display_name, role, department, access_status):
    if CURRENT_SESSION["person_id"] is None:
        raise ValueError("No active enrollment session")

    if len(CURRENT_SESSION["embeddings"]) < SAMPLES_REQUIRED:
        raise ValueError("Not enough face samples")

    embeddings = np.array(CURRENT_SESSION["embeddings"])
    centroid = np.mean(embeddings, axis=0)
    centroid /= np.linalg.norm(centroid)

    
    test_emb = centroid.tolist()
    person, dist = recognize_face(test_emb)

    if person is not None:
        raise ValueError("Person already exists")

    pid = CURRENT_SESSION["person_id"]

    _ensure_enrolled_at_header()

    collection.add(
        embeddings=[centroid.tolist()],
        metadatas=[{"person_id": pid}],
        ids=[f"{pid}_0"]
    )

    header_needed = not os.path.exists(PERSONS_CSV)
    with open(PERSONS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(
                ["person_id", "display_name", "role", "department", "access_status", "enrolled_at"]
            )
        writer.writerow([
            pid,
            display_name,
            role,
            department,
            access_status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ])

    CURRENT_SESSION["person_id"] = None
    CURRENT_SESSION["embeddings"] = []

