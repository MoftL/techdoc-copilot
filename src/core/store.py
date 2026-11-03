import json
import os
from contextvars import ContextVar
from functools import lru_cache

import chromadb

from config import CONFIG

WRITE_BATCH_SIZE = 1000

# None means the bundled sample index. Otherwise, the private workspace of one visitor's upload.
_workspace = ContextVar("workspace", default=None)

_sample_clients = {}
_upload_client = None
_workspace_parents = {}


def set_workspace(workspace_id: str | None):
    _workspace.set(workspace_id)


def scoped_name(name: str) -> str:
    workspace_id = _workspace.get()
    return f"{workspace_id}_{name}" if workspace_id else name


def _get_client(name: str):
    global _upload_client
    if _workspace.get():
        # Uploads live in memory only, so they disappear with the workspace.
        if _upload_client is None:
            _upload_client = chromadb.EphemeralClient()
        return _upload_client

    # One database per collection keeps each file small enough to host on GitHub.
    if name not in _sample_clients:
        path = os.path.join(CONFIG["index_dir"], name)
        os.makedirs(path, exist_ok=True)
        _sample_clients[name] = chromadb.PersistentClient(path=path)
    return _sample_clients[name]


def collection_exists(name: str) -> bool:
    if not _workspace.get() and not os.path.isdir(os.path.join(CONFIG["index_dir"], name)):
        return False
    try:
        return _get_client(name).get_collection(name=scoped_name(name)).count() > 0
    except Exception:
        return False


def get_collection(name: str):
    return _get_client(name).get_collection(name=scoped_name(name))


def recreate_collection(name: str):
    """Ingestion always rebuilds, so chunks from a previous document can never linger."""
    client = _get_client(name)
    try:
        client.delete_collection(name=scoped_name(name))
    except Exception:
        pass
    return client.create_collection(name=scoped_name(name), configuration={"hnsw": {"space": "cosine"}})


def add_chunks(collection, ids: list[str], documents: list[str], vectors: list[list[float]], metadatas: list[dict]):
    for start in range(0, len(ids), WRITE_BATCH_SIZE):
        end = start + WRITE_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=vectors[start:end],
            metadatas=metadatas[start:end]
        )


@lru_cache(maxsize=1)
def _load_sample_parents() -> dict:
    with open(CONFIG["parent_storage_path"], "r", encoding="utf-8") as f:
        return json.load(f)


def save_parents(parents: dict):
    workspace_id = _workspace.get()
    if workspace_id:
        _workspace_parents[workspace_id] = parents
        return
    temp_path = f"{CONFIG['parent_storage_path']}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(parents, f)
    os.replace(temp_path, CONFIG["parent_storage_path"])
    _load_sample_parents.cache_clear()


def load_parents() -> dict:
    workspace_id = _workspace.get()
    return _workspace_parents.get(workspace_id, {}) if workspace_id else _load_sample_parents()


def delete_workspace_data(workspace_id: str):
    token = _workspace.set(workspace_id)
    try:
        for name in CONFIG["collections"].values():
            try:
                _get_client(name).delete_collection(name=scoped_name(name))
            except Exception:
                pass
        _workspace_parents.pop(workspace_id, None)
    finally:
        _workspace.reset(token)
