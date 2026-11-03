from config import CONFIG
from core.llm_client import get_embeddings
from core.store import add_chunks, recreate_collection
from ingestion.chunkers import chunk_sliding_window
from ingestion.loader import chunk_id, load_pages, page_key


def ingest_baseline(file_path: str = CONFIG["default_doc_path"], pages: list[dict] | None = None) -> int:
    pages = pages if pages is not None else load_pages(file_path)
    ids, documents, metadatas = [], [], []

    for page in pages:
        chunks = chunk_sliding_window(page["text"], CONFIG["chunk_size"], CONFIG["chunk_overlap"])
        for i, chunk in enumerate(chunks):
            ids.append(chunk_id("base", page_key(page), i))
            documents.append(chunk)
            metadatas.append({"url": page["url"], "title": page["title"], "section": "", "chunk_index": i})

    print(f"[Baseline] Embedding {len(documents)} chunks from {len(pages)} pages...")
    vectors = get_embeddings(documents)

    collection = recreate_collection(CONFIG["collections"]["baseline"])
    add_chunks(collection, ids, documents, vectors, metadatas)
    print("Baseline Ingestion Complete.")
    return len(documents)


if __name__ == "__main__":
    ingest_baseline()
