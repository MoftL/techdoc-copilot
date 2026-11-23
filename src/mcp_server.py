"""MCP server that gives AI copilots (Claude Desktop, Claude Code, VS Code, Cursor) TechDoc's search as tools.

The copilot calls `search_docs` to get cited, structured excerpts from the documentation and answers from them,
instead of relying on what the model remembers. `index_document` lets it load your own docs first.

Run it over stdio:  python src/mcp_server.py
"""
import contextlib
import os
import sys

from pydantic import BaseModel, Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from config import CONFIG
from core.llm_client import has_api_key
from core.store import collection_exists, set_workspace
from core.workspaces import remove_workspace, workspace_exists
from ingestion.knowledge_base import build_knowledge_base, count_page_tokens
from ingestion.loader import parse_uploads
from retrieval.answer import generate_answer, is_declined
from retrieval.pipelines import PIPELINES

SAMPLE_NAME = "FastAPI documentation (bundled sample)"
ALLOWED_EXTENSIONS = {".md", ".markdown", ".txt"}
MAX_TOP_K = 10

# One server process serves one copilot, so the active knowledge base is plain module state.
_state = {"workspace": None, "name": SAMPLE_NAME}


class Source(BaseModel):
    number: int = Field(description="Cite it as [number]")
    title: str
    section: str
    url: str = Field(description="Link to the original page, empty for uploaded files")
    text: str = Field(description="The documentation excerpt itself")
    context: str = Field(description="AI-written note on where this excerpt sits in the document")
    relevance: str


class SearchResult(BaseModel):
    knowledge_base: str
    question: str
    sources: list[Source]


class AskResult(BaseModel):
    knowledge_base: str
    answer: str
    declined: bool = Field(description="True when the documents don't cover the question")
    sources: list[Source]


class IndexResult(BaseModel):
    knowledge_base: str
    pages: int
    chunks: int
    tokens: int
    used_context: bool = Field(description="False when the file was too large for the AI context step")


server = MCPServer(
    name="TechDoc Copilot",
    instructions=(
        "Search technical documentation and answer only from it. Call search_docs before answering questions "
        "about the indexed framework, and cite sources as [number]. If no source answers the question, say so. "
        "Call index_document first to search a local Markdown or text file instead of the bundled FastAPI docs."
    ),
)


@contextlib.contextmanager
def _quiet():
    """Ingestion prints progress, and stdout is the MCP channel over stdio, so it goes to stderr instead."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _require_api_key():
    if not has_api_key():
        raise ToolError("OPENAI_API_KEY is not set. Add it to .env or the copilot's MCP server config.")


def _activate() -> str:
    """Points the store at the active knowledge base. Uploads expire, so this falls back to the sample."""
    if _state["workspace"] and not workspace_exists(_state["workspace"]):
        _state.update(workspace=None, name=SAMPLE_NAME)
    set_workspace(_state["workspace"])
    if not collection_exists(CONFIG["collections"]["contextual"]):
        raise ToolError("The knowledge base has no index yet. Run python src/build_index.py or call index_document.")
    return _state["name"]


def _clean_question(question: str) -> str:
    question = question.strip()[:CONFIG["max_question_chars"]]
    if not question:
        raise ToolError("The question is empty.")
    return question


def _retrieve(question: str, top_k: int) -> list[dict]:
    with _quiet():
        return PIPELINES[CONFIG["default_pipeline"]]["retrieve"](question, top_k=top_k)


def _to_sources(raw: list[dict]) -> list[Source]:
    return [
        Source(
            number=number,
            title=item["title"],
            section=item["section"],
            url=item["url"],
            text=item["text"],
            context=item["context"],
            relevance=item["score"],
        )
        for number, item in enumerate(raw, 1)
    ]


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def search_docs(question: str, top_k: int = CONFIG["final_top_k"]) -> SearchResult:
    """Find the documentation excerpts that best answer a question, with sources to cite.

    Uses hybrid search (vectors + BM25) and a cross-encoder reranker. No answer is generated, so the
    calling model reads the excerpts and writes the answer itself.
    """
    _require_api_key()
    question = _clean_question(question)
    name = _activate()
    top_k = max(1, min(top_k, MAX_TOP_K))
    return SearchResult(knowledge_base=name, question=question, sources=_to_sources(_retrieve(question, top_k)))


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def ask_docs(question: str) -> AskResult:
    """Answer a question using only the documentation, with [number] citations to the sources returned."""
    _require_api_key()
    question = _clean_question(question)
    name = _activate()
    raw = _retrieve(question, CONFIG["final_top_k"])
    answer = generate_answer(question, raw)
    return AskResult(knowledge_base=name, answer=answer, declined=is_declined(answer), sources=_to_sources(raw))


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
def index_document(path: str) -> IndexResult:
    """Index a local Markdown or text file, then make it the knowledge base that search_docs and ask_docs use.

    It stays in memory for this session only. Small files get an AI-written context note per chunk,
    which costs a few OpenAI calls.
    """
    _require_api_key()
    path = os.path.abspath(os.path.expanduser(path))
    extension = os.path.splitext(path)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ToolError(f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are supported.")
    if not os.path.isfile(path):
        raise ToolError(f"File not found: {path}")
    if os.path.getsize(path) > CONFIG["max_upload_mb"] * 1024 * 1024:
        raise ToolError(f"The file is larger than {CONFIG['max_upload_mb']} MB.")

    with open(path, encoding="utf-8", errors="replace") as f:
        pages = parse_uploads([(os.path.basename(path), f.read())])
    tokens = count_page_tokens(pages)
    if tokens == 0:
        raise ToolError("The file has no text to index.")
    if tokens > CONFIG["max_upload_tokens"]:
        raise ToolError(f"The file has {tokens:,} tokens, more than the {CONFIG['max_upload_tokens']:,} limit.")

    with _quiet():
        result = build_knowledge_base(pages)
    if _state["workspace"]:
        remove_workspace(_state["workspace"])
    _state.update(workspace=result["workspace"], name=os.path.basename(path))

    return IndexResult(
        knowledge_base=_state["name"],
        pages=len(pages),
        chunks=result["chunks"],
        tokens=result["tokens"],
        used_context=result["used_context"],
    )


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
def use_sample_docs() -> str:
    """Switch back to the bundled FastAPI documentation and free the indexed file."""
    if _state["workspace"]:
        remove_workspace(_state["workspace"])
    _state.update(workspace=None, name=SAMPLE_NAME)
    set_workspace(None)
    return f"Now searching the {SAMPLE_NAME}."


if __name__ == "__main__":
    server.run()
