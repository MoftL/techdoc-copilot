import threading
import time
import uuid

from config import CONFIG
from core.store import delete_workspace_data
from retrieval.hybrid_search import clear_bm25_cache

_lock = threading.Lock()
_created_at = {}


def _remove(workspace_id: str):
    _created_at.pop(workspace_id, None)
    delete_workspace_data(workspace_id)
    clear_bm25_cache(workspace_id)


def create_workspace() -> str:
    """Also removes expired and excess workspaces, which keeps memory bounded on a small host."""
    with _lock:
        now = time.time()
        for workspace_id, created in list(_created_at.items()):
            if now - created > CONFIG["workspace_ttl_seconds"]:
                _remove(workspace_id)
        while len(_created_at) >= CONFIG["max_workspaces"]:
            _remove(min(_created_at, key=_created_at.get))

        workspace_id = uuid.uuid4().hex
        _created_at[workspace_id] = now
        return workspace_id


def workspace_exists(workspace_id: str) -> bool:
    with _lock:
        created = _created_at.get(workspace_id)
        return created is not None and time.time() - created <= CONFIG["workspace_ttl_seconds"]


def remove_workspace(workspace_id: str):
    with _lock:
        _remove(workspace_id)
