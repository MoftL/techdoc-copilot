"""The MCP server, tested through a real MCP client connected in-process. Nothing here calls OpenAI."""
import asyncio
import hashlib

import pytest
from mcp import Client

import core.llm_client as llm
import ingestion.baseline_ingest as baseline_ingest
import ingestion.contextual_ingest as contextual_ingest
import ingestion.parent_child_ingest as parent_child_ingest
import mcp_server
import retrieval.advanced_retrieval as advanced_retrieval
from core.store import set_workspace

DOC = """# Zorblax Guide

## Installing Zorblax

Run `zorb install --fast` to install Zorblax. The flux capacitor must be calibrated first.

## Configuring the Quantum Router

Set QUANTUM_ROUTES=7 in zorb.toml to enable the quantum router for every request.
"""


def fake_embedding(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255 + 0.01 for byte in digest[:16]]


def fake_embeddings(texts: list[str]) -> list[list[float]]:
    return [fake_embedding(text) for text in texts]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(mcp_server, "has_api_key", lambda: True)
    for module in (baseline_ingest, parent_child_ingest, contextual_ingest):
        monkeypatch.setattr(module, "get_embeddings", fake_embeddings)
    monkeypatch.setattr("ingestion.chunkers.get_embeddings", fake_embeddings)
    monkeypatch.setattr(contextual_ingest, "generate_text", lambda **kwargs: "Fake context.")
    monkeypatch.setattr(advanced_retrieval, "get_embedding", fake_embedding)
    # The real reranker downloads an ONNX model, so keep the hybrid search order instead.
    monkeypatch.setattr(advanced_retrieval, "rerank_chunks",
                        lambda question, candidate_chunks, top_k: candidate_chunks[:top_k])
    monkeypatch.setattr(mcp_server, "generate_answer", lambda question, sources: "Run `zorb install --fast` [1].")
    yield
    mcp_server.use_sample_docs()
    set_workspace(None)


@pytest.fixture
def doc_path(tmp_path):
    path = tmp_path / "zorblax.md"
    path.write_text(DOC, encoding="utf-8")
    return str(path)


def call(tool: str, arguments: dict):
    async def run():
        async with Client(mcp_server.server) as client:
            return await client.call_tool(tool, arguments)
    return asyncio.run(run())


def call_all(steps: list[tuple[str, dict]]):
    """Several calls in one session, the way a copilot uses the server."""
    async def run():
        async with Client(mcp_server.server) as client:
            return [await client.call_tool(tool, arguments) for tool, arguments in steps]
    return asyncio.run(run())


def test_lists_all_tools_with_output_schemas():
    async def run():
        async with Client(mcp_server.server) as client:
            return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in asyncio.run(run())}
    assert set(tools) == {"search_docs", "ask_docs", "index_document", "use_sample_docs"}
    assert tools["search_docs"].output_schema is not None
    assert tools["search_docs"].annotations.read_only_hint is True
    assert tools["index_document"].annotations.read_only_hint is False


def test_indexed_document_is_searchable_with_numbered_sources(doc_path):
    indexed, searched = call_all([
        ("index_document", {"path": doc_path}),
        ("search_docs", {"question": "QUANTUM_ROUTES zorb.toml", "top_k": 2}),
    ])

    assert not indexed.is_error
    assert indexed.structured_content["knowledge_base"] == "zorblax.md"
    assert indexed.structured_content["chunks"] > 0

    assert not searched.is_error
    result = searched.structured_content
    assert result["knowledge_base"] == "zorblax.md"
    assert [source["number"] for source in result["sources"]] == list(range(1, len(result["sources"]) + 1))
    assert 1 <= len(result["sources"]) <= 2
    assert "QUANTUM_ROUTES" in "\n".join(source["text"] for source in result["sources"])


def test_ask_docs_returns_the_answer_with_its_sources(doc_path):
    _, asked = call_all([
        ("index_document", {"path": doc_path}),
        ("ask_docs", {"question": "How do I install Zorblax?"}),
    ])

    assert not asked.is_error
    assert asked.structured_content["answer"].endswith("[1].")
    assert asked.structured_content["declined"] is False
    assert asked.structured_content["sources"]


def test_switching_back_to_sample_frees_the_upload(doc_path):
    indexed, switched = call_all([
        ("index_document", {"path": doc_path}),
        ("use_sample_docs", {}),
    ])
    assert not switched.is_error
    assert mcp_server._state["workspace"] is None
    assert mcp_server._state["name"] == mcp_server.SAMPLE_NAME


@pytest.mark.parametrize("arguments, message", [
    ({"path": "notes.pdf"}, "supported"),
    ({"path": "missing-file.md"}, "not found"),
])
def test_index_document_rejects_bad_files_with_a_clear_message(arguments, message):
    result = call("index_document", arguments)
    assert result.is_error
    assert message in result.content[0].text


def test_empty_question_is_rejected_with_a_clear_message():
    result = call("search_docs", {"question": "   "})
    assert result.is_error
    assert "empty" in result.content[0].text


def test_missing_api_key_is_explained(monkeypatch):
    monkeypatch.setattr(mcp_server, "has_api_key", lambda: False)
    result = call("search_docs", {"question": "anything"})
    assert result.is_error
    assert "OPENAI_API_KEY" in result.content[0].text


def test_indexing_keeps_stdout_clean_for_the_mcp_channel(doc_path, capsys):
    """Over stdio, anything printed to stdout would corrupt the protocol."""
    mcp_server.index_document(doc_path)
    assert capsys.readouterr().out == ""
