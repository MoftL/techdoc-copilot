from config import CONFIG
from core.llm_client import get_embedding
from core.store import get_collection, load_parents


def retrieve_parent_child(question: str, top_k: int = CONFIG["final_top_k"]) -> list[dict]:
    collection = get_collection(CONFIG["collections"]["parent_child"])
    # Several children often share a parent, so fetch extra to still end up with top_k distinct parents.
    n_results = min(CONFIG["hybrid_fetch_k"], collection.count())
    results = collection.query(query_embeddings=[get_embedding(question)], n_results=n_results)
    parent_storage = load_parents()

    sources, seen = [], set()
    for meta, child_text, distance in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
        parent_id = meta["parent_id"]
        if parent_id in seen or parent_id not in parent_storage:
            continue
        seen.add(parent_id)
        parent = parent_storage[parent_id]
        sources.append({
            "id": parent_id,
            "text": parent["text"],
            "url": parent["url"],
            "title": parent["title"],
            "section": parent["section"],
            "score": f"Similarity {1 - distance:.2f}",
            "matched_text": child_text
        })
        if len(sources) == top_k:
            break
    return sources
