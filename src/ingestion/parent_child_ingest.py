from config import CONFIG
from core.llm_client import get_embeddings
from core.store import add_chunks, recreate_collection, save_parents
from ingestion.chunkers import chunk_by_sections, chunk_sliding_window
from ingestion.loader import chunk_id, load_pages, page_key


def ingest_parent_child(file_path: str = CONFIG["default_doc_path"], pages: list[dict] | None = None) -> int:
    pages = pages if pages is not None else load_pages(file_path)
    parent_storage = {}
    ids, documents, metadatas = [], [], []

    for page in pages:
        for p_idx, parent in enumerate(chunk_by_sections(page["text"], CONFIG["parent_chunk_size"])):
            parent_id = chunk_id("parent", page_key(page), p_idx)
            parent_storage[parent_id] = {
                "text": parent["text"],
                "url": page["url"],
                "title": page["title"],
                "section": parent["section"]
            }
            children = chunk_sliding_window(parent["text"], CONFIG["child_chunk_size"], CONFIG["child_chunk_overlap"])
            for c_idx, child_text in enumerate(children):
                ids.append(f"{parent_id}_child_{c_idx}")
                documents.append(child_text)
                metadatas.append({"url": page["url"], "title": page["title"], "parent_id": parent_id})

    print(f"[Parent-Child] Embedding {len(documents)} children of {len(parent_storage)} parents...")
    vectors = get_embeddings(documents)

    collection = recreate_collection(CONFIG["collections"]["parent_child"])
    add_chunks(collection, ids, documents, vectors, metadatas)

    # Children and parents are replaced together, so a child can never point at a missing or wrong parent.
    save_parents(parent_storage)

    print("Parent-Child Ingestion Complete.")
    return len(documents)


if __name__ == "__main__":
    ingest_parent_child()
