import chromadb
from backend.config import CHROMA_PATH, THRESHOLD
import numpy as np
import csv
import os
import cv2

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("face_embeddings")

PERSONS_CSV = os.path.join(os.path.dirname(CHROMA_PATH), "persons.csv")

def load_persons():
    persons = {}
    if not os.path.exists(PERSONS_CSV):
        return persons
    with open(PERSONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            persons[row["person_id"]] = row
    return persons

def recognize_face(embedding):
    persons_lookup = load_persons()
    res = collection.query(query_embeddings=[embedding], n_results=1)
    if not res["ids"] or not res["metadatas"][0]:
        return None, None
    pid = res["metadatas"][0][0]["person_id"]
    distance = res["distances"][0][0]
    if distance > THRESHOLD:
        return None, distance
    person = persons_lookup.get(pid)
    return person, distance
