import re

from rank_bm25 import BM25Okapi

from core.store import get_collection, scoped_name

TOKEN = re.compile(r"\w+")

_bm25_cache = {}


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def reciprocal_rank_fusion(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[str]:
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc_id in enumerate(sparse_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)


def clear_bm25_cache(workspace_id: str):
    for key in [key for key in _bm25_cache if key.startswith(f"{workspace_id}_")]:
        del _bm25_cache[key]


def _load_keyword_index(collection_name: str) -> dict:
    """Built once per workspace. The corpus includes titles and LLM context, as in Contextual Retrieval."""
    cache_key = scoped_name(collection_name)
    if cache_key not in _bm25_cache:
        data = get_collection(collection_name).get(include=["documents", "metadatas"])
        corpus = [
            tokenize(f"{meta.get('title', '')} {meta.get('section', '')} {meta.get('context', '')} {doc}")
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
        _bm25_cache[cache_key] = {
            "ids": data["ids"],
            "bm25": BM25Okapi(corpus),
            "lookup": {
                doc_id: {"id": doc_id, "text": doc, "metadata": meta}
                for doc_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
            }
        }
    return _bm25_cache[cache_key]


def hybrid_retrieve(
    collection_name: str,
    raw_question: str,
    query_vector: list[float],
    top_k: int = 30,
    use_bm25: bool = True
) -> list[dict]:
    index = _load_keyword_index(collection_name)
    n_fetch = min(top_k, len(index["ids"]))
    if n_fetch == 0:
        return []

    dense_results = get_collection(collection_name).query(query_embeddings=[query_vector], n_results=n_fetch)
    dense_ids = dense_results["ids"][0]
    if not use_bm25:
        return [index["lookup"][doc_id] for doc_id in dense_ids]

    scores = index["bm25"].get_scores(tokenize(raw_question))
    sparse_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_fetch]
    sparse_ids = [index["ids"][i] for i in sparse_indices]

    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)[:n_fetch]
    return [index["lookup"][doc_id] for doc_id in fused_ids]
