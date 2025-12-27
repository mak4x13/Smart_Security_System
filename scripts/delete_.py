import chromadb

client = chromadb.PersistentClient(
    path="backend/db/chromadb_data"
)

collection = client.get_collection("face_embeddings")

collection.delete(
    where={"person_id": "PID_ae711752"}
)

print("Person deleted successfully")
