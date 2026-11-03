import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


CONFIG = {
    "index_dir": os.path.join(ROOT_DIR, "index"),
    "parent_storage_path": os.path.join(ROOT_DIR, "index", "parent_storage.json"),
    "raw_doc_path": os.path.join(ROOT_DIR, "data", "fastapi_doc.md"),
    "default_doc_path": os.path.join(ROOT_DIR, "data", "fastapi_en.md"),
    "eval_questions_path": os.path.join(ROOT_DIR, "eval", "questions.json"),
    "eval_results_path": os.path.join(ROOT_DIR, "eval", "results.json"),

    # Collections for each strategy
    "collections": {
        "baseline": "tech_docs_baseline",
        "parent_child": "tech_docs_children",
        "contextual": "tech_docs_contextual"
    },

    # Models
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",

    # Chunking settings
    "chunk_size": 400,
    "chunk_overlap": 50,
    "parent_chunk_size": 800,
    "child_chunk_size": 150,
    "child_chunk_overlap": 20,
    "semantic_threshold": 0.5,
    "semantic_min_tokens": 120,

    # Retrieval settings
    "hybrid_fetch_k": 30,
    "final_top_k": 4,
    "default_pipeline": "hybrid",

    # Public deployment limits
    "max_question_chars": 500,
    "max_questions_per_session": _env_int("MAX_QUESTIONS_PER_SESSION", 20),
    "max_questions_per_day": _env_int("MAX_QUESTIONS_PER_DAY", 300),
    "max_upload_mb": 10,
    "max_upload_tokens": _env_int("MAX_UPLOAD_TOKENS", 500_000),
    "context_token_limit": _env_int("CONTEXT_TOKEN_LIMIT", 40_000),
    "max_uploads_per_session": _env_int("MAX_UPLOADS_PER_SESSION", 3),
    "max_uploads_per_day": _env_int("MAX_UPLOADS_PER_DAY", 30),

    # Uploaded documents stay in memory for a limited time
    "workspace_ttl_seconds": 2 * 60 * 60,
    "max_workspaces": _env_int("MAX_WORKSPACES", 3),
}
