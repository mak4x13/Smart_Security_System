import csv
from backend.enrollment import collection  # chroma collection

PERSON_CSV = "backend/db/persons.csv"

def list_persons():
    persons = []

    try:
        with open(PERSON_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                persons.append(row)
    except FileNotFoundError:
        pass

    return persons


def delete_person(person_id: str):
    # 1. Delete from ChromaDB
    collection.delete(where={"person_id": person_id})

    # 2. Delete from CSV
    rows = []
    with open(PERSON_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["person_id"] != person_id:
                rows.append(row)

    if rows:
        with open(PERSON_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
