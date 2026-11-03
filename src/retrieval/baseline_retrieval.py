from config import CONFIG
from core.llm_client import get_embedding
from core.store import get_collection


def retrieve_baseline(question: str, top_k: int = CONFIG["final_top_k"]) -> list[dict]:
    collection = get_collection(CONFIG["collections"]["baseline"])
    results = collection.query(query_embeddings=[get_embedding(question)], n_results=min(top_k, collection.count()))

    sources = []
    for chunk_id, text, meta, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        sources.append({
            "id": chunk_id,
            "text": text,
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "score": f"Similarity {1 - distance:.2f}"
        })
    return sources
