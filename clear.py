from config import Config
import shutil

def cleanup():
    vector_store_path = Config.get_vector_store_path()
    if vector_store_path.exists():
        shutil.rmtree(vector_store_path, ignore_errors=True)
    vector_store_path.mkdir(parents=True)
    print("Cleanup complete - ready for fresh initialization")

if __name__ == "__main__":
    cleanup()