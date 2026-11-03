"""Private upload workspaces, tested with fake embeddings. Nothing here calls OpenAI."""
import hashlib

import pytest

import core.llm_client as llm
import ingestion.baseline_ingest as baseline_ingest
import ingestion.contextual_ingest as contextual_ingest
import ingestion.knowledge_base as knowledge_base
import ingestion.parent_child_ingest as parent_child_ingest
import retrieval.advanced_retrieval as advanced_retrieval
import retrieval.baseline_retrieval as baseline_retrieval
import retrieval.parent_child_retrieval as parent_child_retrieval
from config import CONFIG
from core.store import set_workspace
from core.workspaces import create_workspace, remove_workspace, workspace_exists
from ingestion.loader import parse_uploads
from retrieval.hybrid_search import _bm25_cache


def fake_embedding(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255 + 0.01 for byte in digest[:16]]


def fake_embeddings(texts: list[str]) -> list[list[float]]:
    return [fake_embedding(text) for text in texts]


def no_paid_calls(*args, **kwargs):
    raise AssertionError("Paid API call attempted")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(llm, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(llm, "generate_text", no_paid_calls)
    for module in (baseline_ingest, parent_child_ingest, contextual_ingest):
        monkeypatch.setattr(module, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(contextual_ingest, "generate_text", lambda **kwargs: "Fake context.")
    for module in (baseline_retrieval, parent_child_retrieval, advanced_retrieval):
        monkeypatch.setattr(module, "get_embedding", fake_embedding)
    monkeypatch.setattr("ingestion.chunkers.get_embeddings", fake_embeddings)
    yield
    set_workspace(None)


def build(text: str, name: str = "doc.md") -> dict:
    return knowledge_base.build_knowledge_base(parse_uploads([(name, text)]))


def all_text(retrieve, question: str = "anything") -> str:
    return "\n".join(source["text"] for source in retrieve(question))


def keyword_search(question: str) -> list[dict]:
    return advanced_retrieval.retrieve_advanced(question, use_hyde=False, use_rerank=False)


def test_visitors_cannot_see_each_others_documents():
    first = build("# Alpha\n\nThe alphaframe router handles requests.")
    second = build("# Beta\n\nThe betaframe scheduler runs jobs.")

    for retrieve in (baseline_retrieval.retrieve_baseline, parent_child_retrieval.retrieve_parent_child, keyword_search):
        set_workspace(first["workspace"])
        assert "alphaframe" in all_text(retrieve)
        assert "betaframe" not in all_text(retrieve)
        set_workspace(second["workspace"])
        assert "betaframe" in all_text(retrieve)
        assert "alphaframe" not in all_text(retrieve)


def test_very_short_document_still_answers():
    result = build("Tiny doc.")
    assert result["chunks"] == 1
    assert len(baseline_retrieval.retrieve_baseline("q")) == 1
    assert len(parent_child_retrieval.retrieve_parent_child("q")) == 1
    assert len(keyword_search("tiny")) == 1


def test_files_sharing_a_title_do_not_collide():
    pages = parse_uploads([("a.md", "# Guide\n\nFirst file."), ("b.md", "# Guide\n\nSecond file.")])
    result = knowledge_base.build_knowledge_base(pages)
    assert result["chunks"] == 2
    assert {"First file", "Second file"} <= {
        phrase for phrase in ("First file", "Second file") if phrase in all_text(keyword_search)
    }


def test_plain_text_without_headings_uses_the_file_name():
    pages = parse_uploads([("notes.txt", "Just some plain notes.\n\nWith two paragraphs.")])
    assert pages[0]["title"] == "notes.txt"
    knowledge_base.build_knowledge_base(pages)
    assert keyword_search("notes")[0]["title"] == "notes.txt"


def test_suggested_questions_come_from_headings():
    pages = parse_uploads([("a.md", "# Title\n\n## Routing\n\nText.\n\n## Recap\n\nText.\n\n### `Tempo` setting\n\nText.")])
    assert knowledge_base.suggest_questions(pages) == [
        "What do the docs say about Routing?",
        "What do the docs say about Tempo setting?"
    ]
    assert knowledge_base.suggest_questions(parse_uploads([("b.txt", "No headings here.")])) == []


def test_empty_upload_produces_no_pages():
    assert parse_uploads([("empty.md", "   \n\n ")]) == []


def test_large_uploads_skip_the_expensive_context_step(monkeypatch):
    monkeypatch.setattr(contextual_ingest, "generate_text", no_paid_calls)
    monkeypatch.setitem(CONFIG, "context_token_limit", 5)
    result = build("# Big\n\n" + "A long paragraph about things. " * 40)
    assert result["used_context"] is False
    assert result["chunks"] >= 1


def test_small_uploads_get_context():
    result = build("# Small\n\nA short paragraph about routers.")
    assert result["used_context"] is True
    assert keyword_search("routers")[0]["context"] == "Fake context."


def test_removing_a_workspace_deletes_its_data():
    result = build("# Gone\n\nThis will be deleted.")
    keyword_search("deleted")
    workspace = result["workspace"]
    assert any(key.startswith(workspace) for key in _bm25_cache)

    remove_workspace(workspace)
    assert not workspace_exists(workspace)
    assert not any(key.startswith(workspace) for key in _bm25_cache)
    with pytest.raises(Exception):
        baseline_retrieval.retrieve_baseline("q")


def test_oldest_workspace_is_removed_at_capacity(monkeypatch):
    monkeypatch.setitem(CONFIG, "max_workspaces", 2)
    first, second = create_workspace(), create_workspace()
    third = create_workspace()
    remaining = [workspace_exists(w) for w in (first, second, third)]
    assert remaining.count(True) == 2
    assert workspace_exists(third)


def test_expired_workspace_is_reported_gone(monkeypatch):
    workspace = create_workspace()
    monkeypatch.setitem(CONFIG, "workspace_ttl_seconds", -1)
    assert not workspace_exists(workspace)
