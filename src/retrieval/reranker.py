from config import CONFIG

# Measured on the 30 longest chunks: the default batch size peaked at 2011 MB and these settings at 299 MB.
RERANK_BATCH_SIZE = 4
RERANK_MAX_CHARS = 1500
RERANK_THREADS = 2

_model_cache = {}


def get_reranker_model(model_name: str = CONFIG["reranker_model"]):
    """Loads and caches the ONNX Cross-Encoder. Imported here so the UI starts instantly."""
    if model_name not in _model_cache:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        print(f"Loading re-ranker model '{model_name}'...")
        _model_cache[model_name] = TextCrossEncoder(model_name=model_name, threads=RERANK_THREADS)
    return _model_cache[model_name]


def rerank_chunks(
    question: str,
    candidate_chunks: list[dict],
    top_k: int = 3,
    model_name: str = CONFIG["reranker_model"]
) -> list[dict]:
    """
    Reranks candidate chunks based on joint attention cross-encoder scores.
    Expects candidate_chunks to be a list of dicts with at least a 'text' key.
    """
    if not candidate_chunks:
        return []

    model = get_reranker_model(model_name)
    documents = [chunk["text"][:RERANK_MAX_CHARS] for chunk in candidate_chunks]
    scores = list(model.rerank(question, documents, batch_size=RERANK_BATCH_SIZE))

    # Copies, because candidates come from the shared keyword index cache
    ranked_chunks = [{**chunk, "rerank_score": float(score)} for chunk, score in zip(candidate_chunks, scores)]
    ranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked_chunks[:top_k]
