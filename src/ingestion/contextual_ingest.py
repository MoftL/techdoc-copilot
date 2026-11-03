from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from config import CONFIG
from core.llm_client import generate_text, get_embeddings
from core.store import add_chunks, recreate_collection
from ingestion.chunkers import chunk_semantic
from ingestion.loader import chunk_id, load_pages, page_key

PAGE_EXCERPT_CHARS = 6000
MAX_WORKERS = 8


def generate_chunk_context(page: dict, section: str, chunk_text: str) -> str:
    system_prompt = (
        "You write short retrieval context for chunks of technical documentation. "
        "Reply with 1-2 plain sentences naming what the chunk covers and how it fits in the page. "
        "Mention the feature, class and function names involved. Do not start with 'This chunk'."
    )
    user_prompt = (
        f"PAGE TITLE: {page['title']}\nSECTION: {section}\n\n"
        f"START OF PAGE:\n{page['text'][:PAGE_EXCERPT_CHARS]}\n\nCHUNK:\n{chunk_text}"
    )
    try:
        return generate_text(system_prompt=system_prompt, user_prompt=user_prompt).strip()
    except Exception as e:
        # One failed call should not throw away a long, paid-for ingestion run.
        print(f"[Contextual] Context skipped for '{section}': {e}")
        return ""


def enrich_text(title: str, section: str, context: str, chunk: str) -> str:
    return f"{title} > {section}\n{context}\n\n{chunk}"


def ingest_contextual(
    file_path: str = CONFIG["default_doc_path"],
    pages: list[dict] | None = None,
    use_context: bool = True
) -> int:
    pages = pages if pages is not None else load_pages(file_path)

    items = []
    for page in tqdm(pages, desc="Semantic Chunking"):
        chunks = chunk_semantic(
            page["text"],
            similarity_threshold=CONFIG["semantic_threshold"],
            max_tokens=CONFIG["chunk_size"],
            min_tokens=CONFIG["semantic_min_tokens"]
        )
        for i, chunk in enumerate(chunks):
            items.append({"page": page, "index": i, "text": chunk["text"], "section": chunk["section"]})

    if use_context:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            contexts = list(tqdm(
                pool.map(lambda item: generate_chunk_context(item["page"], item["section"], item["text"]), items),
                total=len(items),
                desc="Contextualizing Chunks"
            ))
    else:
        contexts = [""] * len(items)

    enriched = [
        enrich_text(item["page"]["title"], item["section"], context, item["text"])
        for item, context in zip(items, contexts)
    ]
    print(f"[Contextual] Embedding {len(enriched)} contextual chunks...")
    vectors = get_embeddings(enriched)

    collection = recreate_collection(CONFIG["collections"]["contextual"])
    add_chunks(
        collection,
        ids=[chunk_id("context", page_key(item["page"]), item["index"]) for item in items],
        documents=[item["text"] for item in items],
        vectors=vectors,
        metadatas=[
            {"url": item["page"]["url"], "title": item["page"]["title"], "section": item["section"], "context": context}
            for item, context in zip(items, contexts)
        ]
    )
    print("Contextual Ingestion Complete.")
    return len(items)


if __name__ == "__main__":
    ingest_contextual()
