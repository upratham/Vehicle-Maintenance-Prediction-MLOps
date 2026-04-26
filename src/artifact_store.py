"""MongoDB-backed artifact store for model_registry and baselines.

Keeps only the latest MAX_ARTIFACTS documents per profile.
Falls back gracefully if MongoDB is unreachable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.logger import logging

MAX_ARTIFACTS = 5
_REGISTRY_COLLECTION  = "artifact_model_registry"
_BASELINES_COLLECTION = "artifact_baselines"


def _client():
    from src.configuration.mongo_db_connection import MongoDBClient
    return MongoDBClient()


def _save(collection_name: str, profile_name: str, document: dict) -> None:
    try:
        db = _client().database
        col = db[collection_name]
        document = dict(document)
        document["profile_name"] = profile_name
        document["created_at"]   = datetime.now(timezone.utc).isoformat()
        col.insert_one(document)

        # Keep only latest MAX_ARTIFACTS per profile
        ids = [
            d["_id"]
            for d in col.find(
                {"profile_name": profile_name},
                {"_id": 1}
            ).sort("created_at", -1).skip(MAX_ARTIFACTS)
        ]
        if ids:
            col.delete_many({"_id": {"$in": ids}})
            logging.info(f"[artifact_store] Pruned {len(ids)} old docs from {collection_name}/{profile_name}")

        logging.info(f"[artifact_store] Saved to {collection_name}/{profile_name}")
    except Exception as e:
        logging.warning(f"[artifact_store] Save to {collection_name} failed: {e}")


def _load_latest(collection_name: str, profile_name: str | None = None) -> dict | None:
    try:
        db = _client().database
        col = db[collection_name]
        query = {"profile_name": profile_name} if profile_name else {}
        doc = col.find_one(query, {"_id": 0}, sort=[("created_at", -1)])
        return doc
    except Exception as e:
        logging.warning(f"[artifact_store] Load from {collection_name} failed: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def save_model_registry(profile_name: str, manifest: dict) -> None:
    _save(_REGISTRY_COLLECTION, profile_name, manifest)


def load_model_registry(profile_name: str | None = None) -> dict | None:
    return _load_latest(_REGISTRY_COLLECTION, profile_name)


def save_baselines(profile_name: str, baselines: dict) -> None:
    _save(_BASELINES_COLLECTION, profile_name, baselines)


def load_baselines(profile_name: str | None = None) -> dict | None:
    return _load_latest(_BASELINES_COLLECTION, profile_name)
