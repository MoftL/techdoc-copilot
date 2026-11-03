from functools import partial

from config import CONFIG
from retrieval.advanced_retrieval import retrieve_advanced
from retrieval.baseline_retrieval import retrieve_baseline
from retrieval.parent_child_retrieval import retrieve_parent_child

PIPELINES = {
    "hybrid": {
        "name": "Hybrid + Reranker",
        "summary": "Vector and keyword search over contextual chunks, then a cross-encoder reranker. The best result for its speed.",
        "collection": CONFIG["collections"]["contextual"],
        "retrieve": partial(retrieve_advanced, use_hyde=False)
    },
    "advanced": {
        "name": "Advanced (HyDE)",
        "summary": "Adds HyDE to Hybrid + Reranker. Slower, and it did not improve accuracy in our evaluation.",
        "collection": CONFIG["collections"]["contextual"],
        "retrieve": retrieve_advanced
    },
    "parent_child": {
        "name": "Parent-Child",
        "summary": "Searches small, precise chunks and returns the larger section around them.",
        "collection": CONFIG["collections"]["parent_child"],
        "retrieve": retrieve_parent_child
    },
    "baseline": {
        "name": "Baseline",
        "summary": "Plain vector search over fixed-size chunks. The benchmark for everything else.",
        "collection": CONFIG["collections"]["baseline"],
        "retrieve": retrieve_baseline
    }
}
