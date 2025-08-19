# utils/brand_utils.py
import json
from pathlib import Path
from typing import Dict, Optional, Any
import os


BRANDS_FILE = Path("data/brands.json")

def _ensure_brands_file():
    """Create brands file if it doesn't exist"""
    BRANDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not BRANDS_FILE.exists():
        BRANDS_FILE.write_text(json.dumps({}))

def _load_brands() -> Dict[str, Dict]:
    """Load all brands from file, converting old list format if needed"""
    _ensure_brands_file()
    try:
        data = json.loads(BRANDS_FILE.read_text())
        
        # Convert old list format to new dictionary format
        if isinstance(data, list):
            new_data = {brand['key']: {k: v for k, v in brand.items() if k != 'key'} 
                       for brand in data}
            _save_brands(new_data)  # Migrate to new format
            return new_data
            
        return data
    except json.JSONDecodeError:
        return {}

def _save_brands(brands: Dict[str, Dict]):
    """Save brands in dictionary format"""
    _ensure_brands_file()
    with open(BRANDS_FILE, 'w') as f:
        json.dump(brands, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

def add_brand(key: str, config: Dict):
    """Add or update a brand"""
    brands = _load_brands()
    brands[key] = config
    _save_brands(brands)

def get_brand_config(brand_key: str) -> Dict[str, Any]:
    """Load brand configuration with proper string formatting"""
    with open(BRANDS_FILE, 'r', encoding='utf-8') as f:
        brands = json.load(f)
    
    if brand_key not in brands:
        raise ValueError(f"Brand {brand_key} not found in configuration")
    
    config = brands[brand_key].copy()
    
    # Format system message (no JSON corrections)
    config["system_message"] = config["system_message"].format(
        display_name=config["display_name"],
        word_limit=config["word_limit"],
        off_topic_response=config["off_topic_response"],
        corrections=config["corrections"]
    )
    
    # Format prompt template
    config["prompt_template"] = config["prompt_template"].format(
        display_name=config["display_name"],
        context="{context}",
        question="{question}",
        conversation_history="{conversation_history}"
    )
    
    return config

def get_all_brands() -> Dict[str, Dict]:
    """Get all brands with their configurations"""
    return _load_brands()

def update_brand(key: str, config: Dict):
    """Update an existing brand"""
    brands = _load_brands()
    if key not in brands:
        raise ValueError("Brand does not exist")
    brands[key] = config
    _save_brands(brands)

def delete_brand(key: str):
    """Delete a brand"""
    try:
        brands = _load_brands()
        if key not in brands:
            raise ValueError("Brand does not exist")
        
        # Create backup in case of failure
        backup = brands.copy()
        del brands[key]
        
        try:
            _save_brands(brands)
            # Verify write was successful
            if key in _load_brands():
                # Write failed, restore backup
                _save_brands(backup)
                raise RuntimeError("Brand deletion not persisted to storage")
        except Exception as e:
            _save_brands(backup)  # Restore backup on error
            raise RuntimeError(f"Failed to persist deletion: {str(e)}")
            
    except Exception as e:
        raise ValueError(f"Brand deletion failed: {str(e)}")
    
def _save_brands(brands: Dict[str, Dict]):
    """Save brands to file with fsync for reliability"""
    _ensure_brands_file()
    temp_file = BRANDS_FILE.with_suffix('.tmp')
    
    try:
        with open(temp_file, 'w') as f:
            json.dump(brands, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomic replace
        temp_file.replace(BRANDS_FILE)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to save brands: {str(e)}")
    
def brand_exists(brand_key: str) -> bool:
    """Check if a brand with given key exists"""
    from config import Config
    import json
    
    if not Config.BRANDS_FILE.exists():
        return False
        
    with open(Config.BRANDS_FILE, 'r') as f:
        brands = json.load(f)
        return brand_key in brands
    
