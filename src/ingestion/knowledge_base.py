import re
from collections.abc import Callable

from config import CONFIG
from core.store import set_workspace
from core.workspaces import create_workspace, remove_workspace
from ingestion.baseline_ingest import ingest_baseline
from ingestion.chunkers import count_tokens
from ingestion.contextual_ingest import ingest_contextual
from ingestion.parent_child_ingest import ingest_parent_child

SUBHEADING = re.compile(r"(?m)^#{2,3}\s+(.+)$")
SKIPPED_HEADINGS = {"recap", "summary", "overview", "introduction", "conclusion", "next steps", "table of contents"}


def count_page_tokens(pages: list[dict]) -> int:
    return sum(count_tokens(page["text"]) for page in pages)


def suggest_questions(pages: list[dict], limit: int = 4) -> list[str]:
    """Free starter questions taken from the document's own headings, with no LLM call."""
    questions = []
    for page in pages:
        for match in SUBHEADING.finditer(page["text"]):
            heading = re.sub(r"[`*]", "", match.group(1)).strip()
            if len(heading) >= 4 and heading.lower() not in SKIPPED_HEADINGS:
                question = f"What do the docs say about {heading}?"
                if question not in questions:
                    questions.append(question)
            if len(questions) == limit:
                return questions
    return questions


def build_knowledge_base(pages: list[dict], on_step: Callable[[str], None] = lambda _: None) -> dict:
    """Builds all three indexes for one visitor's documents, in a workspace only they can reach."""
    tokens = count_page_tokens(pages)
    # An LLM call per chunk is the only expensive step, so large uploads skip it and stay cheap.
    use_context = tokens <= CONFIG["context_token_limit"]

    workspace_id = create_workspace()
    set_workspace(workspace_id)
    try:
        on_step("Creating fixed-size chunks...")
        ingest_baseline(pages=pages)
        on_step("Creating parent and child chunks...")
        ingest_parent_child(pages=pages)
        on_step("Creating semantic chunks with context..." if use_context else "Creating semantic chunks...")
        chunks = ingest_contextual(pages=pages, use_context=use_context)
    except Exception:
        remove_workspace(workspace_id)
        set_workspace(None)
        raise

    return {"workspace": workspace_id, "chunks": chunks, "tokens": tokens, "used_context": use_context}
