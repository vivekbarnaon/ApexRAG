import os
import pickle
import re

memory_cache = {}
MEMORY_CACHE_SIZE = 100
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _safe_key(key: str) -> str:
    # Allow only alphanumerics, dot, underscore, hyphen; replace others with underscore
    return re.sub(r'[^A-Za-z0-9._-]', '_', key)


def manage_memory_cache():
    if len(memory_cache) > MEMORY_CACHE_SIZE:
        items_to_remove = len(memory_cache) - int(MEMORY_CACHE_SIZE * 0.8)
        for _ in range(items_to_remove):
            memory_cache.pop(next(iter(memory_cache)))


def save_cache(cache_key: str, data):
    memory_cache[cache_key] = data
    manage_memory_cache()
    try:
        safe = _safe_key(cache_key)
        with open(os.path.join(CACHE_DIR, f"{safe}.pkl"), "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass


def load_cache(cache_key: str):
    if cache_key in memory_cache:
        return memory_cache[cache_key]
    safe = _safe_key(cache_key)
    path = os.path.join(CACHE_DIR, f"{safe}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                memory_cache[cache_key] = data
                manage_memory_cache()
                return data
        except Exception:
            pass
    return None

