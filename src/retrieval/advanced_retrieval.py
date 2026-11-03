from config import CONFIG
from core.llm_client import get_embedding
from retrieval.hyde import generate_hyde_vector
from retrieval.hybrid_search import hybrid_retrieve
from retrieval.reranker import rerank_chunks


def retrieve_advanced(
    question: str,
    top_k: int = CONFIG["final_top_k"],
    use_hyde: bool = True,
    use_bm25: bool = True,
    use_rerank: bool = True
) -> list[dict]:
    """The use_* switches let the evaluation measure what each stage adds."""
    query_vector = generate_hyde_vector(question) if use_hyde else get_embedding(question)

    candidates = hybrid_retrieve(
        collection_name=CONFIG["collections"]["contextual"],
        raw_question=question,
        query_vector=query_vector,
        top_k=CONFIG["hybrid_fetch_k"],
        use_bm25=use_bm25
    )

    if use_rerank:
        top_chunks = rerank_chunks(question=question, candidate_chunks=candidates, top_k=top_k)
    else:
        top_chunks = candidates[:top_k]

    sources = []
    for chunk in top_chunks:
        meta = chunk["metadata"]
        score = f"Relevance {chunk['rerank_score']:.2f}" if "rerank_score" in chunk else ""
        sources.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "context": meta.get("context", ""),
            "score": score
        })
    return sources
