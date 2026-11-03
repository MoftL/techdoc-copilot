import argparse
import os
import time

from config import CONFIG
from ingestion.baseline_ingest import ingest_baseline
from ingestion.contextual_ingest import ingest_contextual
from ingestion.loader import extract_english, load_pages
from ingestion.parent_child_ingest import ingest_parent_child

INGESTERS = {
    "baseline": ingest_baseline,
    "parent_child": ingest_parent_child,
    "contextual": ingest_contextual
}


def main():
    parser = argparse.ArgumentParser(description="Build the search indexes from the English FastAPI docs.")
    parser.add_argument("--only", choices=INGESTERS, help="Build a single index instead of all three.")
    args = parser.parse_args()

    os.makedirs(CONFIG["index_dir"], exist_ok=True)
    if not os.path.exists(CONFIG["default_doc_path"]):
        count = extract_english(CONFIG["raw_doc_path"], CONFIG["default_doc_path"])
        print(f"Extracted {count} English pages to {CONFIG['default_doc_path']}")

    pages = load_pages(CONFIG["default_doc_path"])
    for name, ingest in INGESTERS.items():
        if args.only and name != args.only:
            continue
        started = time.time()
        count = ingest(pages=pages)
        print(f"{name}: {count} chunks in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
