# src/ai_core/model_cache.py

import os
import joblib

def _model_path(cache_dir, facility, item, model_name):
    os.makedirs(cache_dir, exist_ok=True)
    safe_facility = str(facility).replace(" ", "_")
    safe_item = str(item).replace(" ", "_")
    return os.path.join(
        cache_dir,
        f"{model_name}_{safe_facility}_{safe_item}.pkl"
    )


def load_cached_model(cache_dir, facility, item, model_name):
    """
    Load cached model if exists.
    Returns (model, True) or (None, False)
    """
    path = _model_path(cache_dir, facility, item, model_name)

    if os.path.exists(path):
        return joblib.load(path), True

    return None, False


def save_cached_model(model, cache_dir, facility, item, model_name):
    """
    Persist trained model to disk
    """
    path = _model_path(cache_dir, facility, item, model_name)
    joblib.dump(model, path)
