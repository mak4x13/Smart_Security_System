# backend/db/create_chroma_db.py

import chromadb

# ---------- Config ----------
PERSIST_DIR = "backend/db/chromadb_data"  # path for persistent ChromaDB
COLLECTION_NAME = "face_embeddings"

# ---------- Create Persistent ChromaDB ----------
def create_chroma_db():
    # Initialize persistent client
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Create collection if it doesn't exist
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except chromadb.errors.NotFoundError:
        collection = client.create_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' created successfully.")

    return collection

if __name__ == "__main__":
    create_chroma_db()
