"""Measures retrieval quality for each pipeline, plus each stage of the advanced pipeline."""
import json
import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import CONFIG
from ingestion.loader import page_path
from retrieval.advanced_retrieval import retrieve_advanced
from retrieval.answer import generate_answer, is_declined
from retrieval.baseline_retrieval import retrieve_baseline
from retrieval.parent_child_retrieval import retrieve_parent_child

VARIANTS = [
    ("Baseline", retrieve_baseline),
    ("Parent-Child", retrieve_parent_child),
    ("Contextual chunks only", partial(retrieve_advanced, use_hyde=False, use_bm25=False, use_rerank=False)),
    ("Contextual + BM25", partial(retrieve_advanced, use_hyde=False, use_bm25=True, use_rerank=False)),
    ("Contextual + BM25 + reranker", partial(retrieve_advanced, use_hyde=False, use_bm25=True, use_rerank=True)),
    ("Advanced (adds HyDE)", retrieve_advanced),
]


def first_correct_rank(sources: list[dict], expected_pages: list[str]) -> int | None:
    for rank, source in enumerate(sources, 1):
        if page_path(source["url"]) in expected_pages:
            return rank
    return None


def evaluate_variant(label: str, retrieve, questions: list[dict]) -> dict:
    ranks, misses, elapsed = [], [], 0.0
    for item in questions:
        started = time.perf_counter()
        sources = retrieve(item["question"])
        elapsed += time.perf_counter() - started
        rank = first_correct_rank(sources, item["pages"])
        ranks.append(rank)
        if rank is None:
            misses.append(item["question"])

    total = len(questions)
    result = {
        "label": label,
        "hit_rate_at_k": sum(r is not None for r in ranks) / total,
        "hit_rate_at_1": sum(r == 1 for r in ranks) / total,
        "mrr": sum(1 / r for r in ranks if r) / total,
        "avg_seconds": elapsed / total,
        "misses": misses
    }
    print(
        f"{label:32} hit@{CONFIG['final_top_k']}={result['hit_rate_at_k']:.3f} "
        f"hit@1={result['hit_rate_at_1']:.3f} mrr={result['mrr']:.3f} {result['avg_seconds']:.2f}s"
    )
    return result


def evaluate_refusals(retrieve, label: str, questions: list[str]) -> dict:
    correct, wrongly_answered = 0, []
    for question in questions:
        answer = generate_answer(question, retrieve(question))
        if is_declined(answer):
            correct += 1
        else:
            wrongly_answered.append(question)
    print(f"Declined {correct}/{len(questions)} out-of-scope questions using {label}")
    return {"pipeline": label, "correct": correct, "total": len(questions), "wrongly_answered": wrongly_answered}


def main():
    with open(CONFIG["eval_questions_path"], "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Warm up first so the model load isn't counted as query time.
    retrieve_advanced("warm up", use_hyde=False)

    pipelines = [evaluate_variant(label, retrieve, questions["answerable"]) for label, retrieve in VARIANTS]
    refusals = evaluate_refusals(retrieve_advanced, "Advanced", questions["out_of_scope"])

    results = {
        "top_k": CONFIG["final_top_k"],
        "question_count": len(questions["answerable"]),
        "pipelines": pipelines,
        "refusals": refusals
    }
    with open(CONFIG["eval_results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {CONFIG['eval_results_path']}")


if __name__ == "__main__":
    main()
