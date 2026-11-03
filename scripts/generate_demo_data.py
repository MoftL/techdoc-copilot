"""Records real pipeline runs for the static demo page, so visitors can explore results without an API key."""
import json
import os
import sys
import time

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from config import CONFIG
from ingestion.chunkers import chunk_by_sections, chunk_semantic, chunk_sliding_window, count_tokens
from ingestion.loader import load_pages
from retrieval.answer import generate_answer
from retrieval.pipelines import PIPELINES

OUTPUT_PATH = os.path.join(ROOT_DIR, "docs", "data", "demo.json")
EXCERPT_CHARS = 700
CHUNKING_PAGE = "https://fastapi.tiangolo.com/advanced/response-headers/"
QUESTIONS = [
    "How do I add custom headers to a response?",
    "How do I return a 404 error when an item does not exist?",
    "How do I run a task after returning a response?",
    "When should I use def instead of async def?",
    "How do I hash passwords and issue JWT tokens?",
    "How do I use a class as a dependency?",
    "How do I center a div with CSS?"
]


def record_question(question: str) -> dict:
    runs = {}
    for key, pipeline in PIPELINES.items():
        started = time.perf_counter()
        sources = pipeline["retrieve"](question)
        seconds = time.perf_counter() - started
        runs[key] = {
            "answer": generate_answer(question, sources),
            "seconds": round(seconds, 2),
            "sources": [
                {
                    "heading": source["section"] or source["title"],
                    "url": source["url"],
                    "score": source.get("score", ""),
                    "context": source.get("context", ""),
                    "excerpt": source["text"][:EXCERPT_CHARS]
                }
                for source in sources
            ]
        }
        print(f"{question[:40]:40} {key:14} {seconds:.2f}s")
    return {"question": question, "runs": runs}


def describe(chunks: list[str]) -> list[dict]:
    return [{"tokens": count_tokens(chunk), "text": chunk} for chunk in chunks]


def record_chunking() -> dict:
    page = next(p for p in load_pages(CONFIG["default_doc_path"]) if p["url"] == CHUNKING_PAGE)
    text = page["text"]
    # What the first version of this project did, kept here to show why it was replaced.
    first_version = [s.strip() + "." for s in text.split(".") if len(s.strip()) > 10]
    parents = chunk_by_sections(text, CONFIG["parent_chunk_size"])
    return {
        "page": CHUNKING_PAGE,
        "strategies": {
            "first_version": describe(first_version),
            "baseline": describe(chunk_sliding_window(text, CONFIG["chunk_size"], CONFIG["chunk_overlap"])),
            "parents": describe([p["text"] for p in parents]),
            "children": describe(
                chunk_sliding_window(parents[0]["text"], CONFIG["child_chunk_size"], CONFIG["child_chunk_overlap"])
            ),
            "semantic": describe([
                c["text"] for c in chunk_semantic(
                    text,
                    similarity_threshold=CONFIG["semantic_threshold"],
                    max_tokens=CONFIG["chunk_size"],
                    min_tokens=CONFIG["semantic_min_tokens"]
                )
            ])
        }
    }


def main():
    with open(CONFIG["eval_results_path"], "r", encoding="utf-8") as f:
        evaluation = json.load(f)

    # Warm up first so loading the reranker isn't counted as search time.
    PIPELINES["hybrid"]["retrieve"]("warm up")

    data = {
        "pipelines": {
            key: {"name": p["name"], "summary": p["summary"]} for key, p in PIPELINES.items()
        },
        "default_pipeline": CONFIG["default_pipeline"],
        "questions": [record_question(question) for question in QUESTIONS],
        "chunking": record_chunking(),
        "evaluation": evaluation
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
