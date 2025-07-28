import chromadb
import json
import shutil
from pathlib import Path
from chromadb.config import Settings

# CONFIG - Update these paths
OLD_STORE = Path("D:/CSPEREZ/PROGRAMS/PythonProjects/ai-agent/data/vector_stores")
NEW_STORE = Path("D:/CSPEREZ/PROGRAMS/PythonProjects/ai-agent/data/vector_stores_v2")
BACKUP_DIR = Path("D:/chroma_backup")

def legacy_client():
    """Create client with legacy settings"""
    return chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(OLD_STORE),
        anonymized_telemetry=False
    ))

def export_data():
    """Export all collections to JSON"""
    BACKUP_DIR.mkdir(exist_ok=True)
    client = legacy_client()
    
    for collection in client.list_collections():
        data = collection.get()
        with open(BACKUP_DIR/f"{collection.name}.json", "w") as f:
            json.dump({
                "ids": data["ids"],
                "documents": data["documents"],
                "metadatas": data["metadatas"],
                "embeddings": data.get("embeddings", [])
            }, f, indent=2)

def create_new_store():
    """Create fresh Chroma 1.x+ store"""
    from chromadb import PersistentClient
    client = PersistentClient(path=str(NEW_STORE))
    
    for json_file in BACKUP_DIR.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        collection = client.get_or_create_collection(json_file.stem)
        collection.add(
            ids=data["ids"],
            documents=data["documents"],
            metadatas=data["metadatas"],
            embeddings=data["embeddings"]
        )

if __name__ == "__main__":
    print("Step 1: Exporting data from old store...")
    export_data()
    
    print("Step 2: Creating new store...")
    create_new_store()
    
    print("Migration complete! New store at:", NEW_STORE)