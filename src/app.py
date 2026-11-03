import json
import logging
import os
import threading
import time
from datetime import date

import streamlit as st

from config import CONFIG
from core.llm_client import has_api_key
from core.store import collection_exists, set_workspace
from core.workspaces import remove_workspace, workspace_exists
from ingestion.knowledge_base import build_knowledge_base, count_page_tokens, suggest_questions
from ingestion.loader import parse_uploads
from retrieval.answer import generate_answer, is_declined, link_citations, stream_answer
from retrieval.pipelines import PIPELINES

logger = logging.getLogger(__name__)

APP_NAME = "TechDoc Copilot"
SAMPLE_DOCS_URL = "https://fastapi.tiangolo.com/"
GENERIC_ERROR = "Something went wrong while answering. Please try again in a moment."
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}
SAMPLE_QUESTIONS = [
    "How do I add custom headers to a response?",
    "What is the difference between Depends and Security?",
    "How do I run a task after returning a response?",
    "How do I upload a file and read its contents?"
]

st.set_page_config(page_title=APP_NAME, page_icon=":material/menu_book:", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 920px; padding-top: 4rem; }
    section[data-testid="stSidebar"] { min-width: 280px; }
    .brand { font-size: 1.1rem; font-weight: 700; letter-spacing: -0.015em; margin-bottom: 0.1rem; }
    .brand-sub { font-size: 0.82rem; opacity: 0.68; margin-bottom: 1.25rem; }
    h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.025em; }
    .hero-title { font-size: clamp(2rem, 5vw, 2.9rem); font-weight: 700; line-height: 1.12; letter-spacing: -0.035em; margin: 0.5rem 0 0.75rem; }
    .hero-title span { color: #0D9488; }
    .hero-text { font-size: 1.05rem; line-height: 1.55; opacity: 0.78; max-width: 640px; margin-bottom: 1.5rem; }
    .steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.75rem; margin: 0.5rem 0 1.75rem; }
    .step { border: 1px solid rgba(128, 128, 128, 0.28); border-radius: 0.6rem; padding: 0.9rem 1rem; background: rgba(128, 128, 128, 0.07); }
    .step-number { font-size: 0.75rem; font-weight: 700; color: #0D9488; margin-bottom: 0.25rem; }
    .step-title { font-weight: 600; margin-bottom: 0.15rem; }
    .step-text { font-size: 0.85rem; line-height: 1.4; opacity: 0.72; }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def get_daily_usage() -> dict:
    """Shared by every visitor, so it protects the API budget even if someone reloads to reset their session."""
    return {"date": date.today(), "questions": 0, "uploads": 0, "lock": threading.Lock()}


def _reserve_daily(kind: str, amount: int, limit: int) -> bool:
    usage = get_daily_usage()
    with usage["lock"]:
        if usage["date"] != date.today():
            usage["date"], usage["questions"], usage["uploads"] = date.today(), 0, 0
        if usage[kind] + amount > limit:
            return False
        usage[kind] += amount
        return True


def questions_left() -> int:
    used = st.session_state.get("questions_used", 0)
    return max(0, CONFIG["max_questions_per_session"] - used)


def spend_questions(amount: int) -> str | None:
    """Reserves budget for a request. Returns the reason if it must be refused."""
    if questions_left() < amount:
        return "You've reached the question limit for this session. Please come back later."
    if not _reserve_daily("questions", amount, CONFIG["max_questions_per_day"]):
        return "This free demo has reached its daily limit. Please come back tomorrow."
    st.session_state.questions_used = st.session_state.get("questions_used", 0) + amount
    return None


def spend_upload() -> str | None:
    if st.session_state.get("uploads_used", 0) >= CONFIG["max_uploads_per_session"]:
        return "You've reached the upload limit for this session. Please come back later."
    if not _reserve_daily("uploads", 1, CONFIG["max_uploads_per_day"]):
        return "This free demo has reached its daily upload limit. Please come back tomorrow."
    st.session_state.uploads_used = st.session_state.get("uploads_used", 0) + 1
    return None


def validate_question(question: str) -> str | None:
    if len(question) > CONFIG["max_question_chars"]:
        return f"Please keep your question under {CONFIG['max_question_chars']} characters."
    return None


def missing_key_message() -> str | None:
    if not has_api_key():
        return "The OpenAI API key is not configured. Add `OPENAI_API_KEY` to `.env` or the app's secrets."
    return None


def sample_available() -> bool:
    return all(collection_exists(p["collection"]) for p in PIPELINES.values())


def reset_conversation():
    st.session_state.messages = []
    st.session_state.pop("comparison", None)


def close_knowledge_base():
    knowledge_base = st.session_state.pop("knowledge_base", None)
    if knowledge_base and knowledge_base["workspace"]:
        remove_workspace(knowledge_base["workspace"])
    set_workspace(None)
    reset_conversation()


def activate_current_knowledge_base() -> dict | None:
    """Runs on every page load, so all retrieval below only ever sees this visitor's documents."""
    knowledge_base = st.session_state.get("knowledge_base")
    if knowledge_base and knowledge_base["workspace"] and not workspace_exists(knowledge_base["workspace"]):
        st.session_state.pop("knowledge_base")
        reset_conversation()
        st.session_state.notice = "Your documents were removed from memory after a while. Please upload them again."
        knowledge_base = None
    set_workspace(knowledge_base["workspace"] if knowledge_base else None)
    return knowledge_base


def read_uploads(uploaded_files: list, pasted_text: str) -> tuple[list[tuple[str, str]], str | None]:
    files = []
    for uploaded in uploaded_files:
        if uploaded.size > CONFIG["max_upload_mb"] * 1024 * 1024:
            return [], f"{uploaded.name} is over the {CONFIG['max_upload_mb']} MB limit."
        text = uploaded.getvalue().decode("utf-8", errors="replace")
        if "\x00" in text:
            return [], f"{uploaded.name} doesn't look like a text file."
        files.append((uploaded.name, text))
    if pasted_text.strip():
        files.append(("Pasted text", pasted_text))
    return files, None


def build_from_uploads(uploaded_files: list, pasted_text: str):
    files, problem = read_uploads(uploaded_files, pasted_text)
    if problem:
        st.warning(problem)
        return
    pages = parse_uploads(files)
    if not pages:
        st.warning("Please upload a file or paste some documentation text first.")
        return

    tokens = count_page_tokens(pages)
    if tokens > CONFIG["max_upload_tokens"]:
        st.warning(
            f"This is about {tokens:,} tokens, and the limit is {CONFIG['max_upload_tokens']:,}. "
            "Try uploading only the sections you need."
        )
        return

    refusal = spend_upload()
    if refusal:
        st.info(refusal)
        return

    close_knowledge_base()
    try:
        with st.status("Building your knowledge base. Large uploads can take a few minutes.", expanded=True) as status:
            result = build_knowledge_base(pages, on_step=st.write)
            status.update(label="Knowledge base ready", state="complete")
    except Exception:
        logger.exception("Failed to build knowledge base")
        st.error("We couldn't process those documents. Please try again.")
        return

    names = [name for name, _ in files]
    st.session_state.knowledge_base = {
        "workspace": result["workspace"],
        "name": names[0] if len(names) == 1 else f"{len(names)} documents",
        "detail": f"{len(pages)} {'page' if len(pages) == 1 else 'pages'} · {result['chunks']} chunks",
        "used_context": result["used_context"],
        "examples": suggest_questions(pages)
    }
    st.rerun()


def setup_screen():
    st.markdown(
        """
        <div class="hero-title">Learn any framework<br><span>from its own docs</span></div>
        <div class="hero-text">Add a framework's documentation and ask questions in plain English.
        Every answer comes only from your documents and shows its sources.</div>
        <div class="steps">
          <div class="step"><div class="step-number">Step 1</div><div class="step-title">Add your docs</div>
            <div class="step-text">Upload Markdown or text files, or paste them in.</div></div>
          <div class="step"><div class="step-number">Step 2</div><div class="step-title">Build a knowledge base</div>
            <div class="step-text">Your docs are split, enriched and indexed for search.</div></div>
          <div class="step"><div class="step-number">Step 3</div><div class="step-title">Ask and verify</div>
            <div class="step-text">Get a direct answer, with the exact sources behind it.</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    notice = st.session_state.pop("notice", None)
    if notice:
        st.info(notice)

    problem = missing_key_message()
    if problem:
        st.warning(problem)

    uploaded_files = st.file_uploader(
        "Upload documentation",
        type=["md", "mdx", "txt", "rst"],
        accept_multiple_files=True,
        help=f"Up to {CONFIG['max_upload_tokens']:,} tokens in total, which is about 2.5 MB of text."
    )
    with st.expander("Or paste text"):
        pasted_text = st.text_area("Documentation text", height=200, label_visibility="collapsed")

    if st.button("Build knowledge base", type="primary", icon=":material/auto_awesome:", disabled=bool(problem)):
        build_from_uploads(uploaded_files, pasted_text)

    st.caption(
        "Your documents are private to this session and are removed from memory within "
        f"{CONFIG['workspace_ttl_seconds'] // 3600} hours. They are sent to OpenAI for processing, "
        "so don't upload anything confidential."
    )

    if sample_available():
        st.divider()
        st.markdown("##### No documents ready?")
        if st.button("Try it with the FastAPI docs", icon=":material/play_arrow:", disabled=bool(problem)):
            st.session_state.knowledge_base = {
                "workspace": None,
                "name": "FastAPI docs",
                "detail": "Sample · 138 pages",
                "used_context": True,
                "examples": SAMPLE_QUESTIONS
            }
            reset_conversation()
            st.rerun()


def render_knowledge_base_panel(knowledge_base: dict):
    with st.sidebar:
        st.markdown(f"**{knowledge_base['name']}**")
        st.caption(knowledge_base["detail"])
        if st.button("Change documents", icon=":material/swap_horiz:", width="stretch"):
            close_knowledge_base()
            st.rerun()


def source_heading(source: dict) -> str:
    """The last part of the heading path is the most specific, and it fits on one line."""
    heading = source["section"] or source["title"] or "Source"
    return heading.split(" > ")[-1].replace("`", "")


def render_sources(sources: list[dict]):
    if not sources:
        return
    st.markdown("**Sources**")
    for number, source in enumerate(sources, 1):
        with st.expander(f"{number} · {source_heading(source)}"):
            details = [d for d in (source["title"] if source["section"] else "", source.get("score", "")) if d]
            if details:
                st.caption(" · ".join(details))
            if source["url"]:
                st.markdown(f"[Open this page in the docs]({source['url']})")
            if source.get("context"):
                st.markdown(f"*{source['context']}*")
            st.code(source["text"], language="markdown", wrap_lines=True)


def answer_footer(pipeline_name: str, source_count: int, seconds: float) -> str:
    return " · ".join([pipeline_name, f"{source_count} sources", f"searched in {seconds:.1f}s"])


def render_answer(question: str, pipeline_key: str) -> dict | None:
    """Retrieves, streams the answer into the page, and returns it for the chat history."""
    pipeline = PIPELINES[pipeline_key]
    try:
        started = time.perf_counter()
        with st.spinner("Searching the documentation..."):
            sources = pipeline["retrieve"](question)
        seconds = time.perf_counter() - started

        placeholder = st.empty()
        answer = ""
        for token in stream_answer(question, sources):
            answer += token
            placeholder.markdown(answer)
        answer = link_citations(answer, sources)
        placeholder.markdown(answer)

        # A declined answer used none of what was searched, so showing it as sources would mislead.
        if is_declined(answer):
            sources = []
        footer = answer_footer(pipeline["name"], len(sources), seconds)
        st.caption(footer)
        render_sources(sources)
        return {"role": "assistant", "content": answer, "footer": footer, "sources": sources}
    except Exception:
        # Details go to the log. Raw errors can expose keys, paths and internals to visitors.
        logger.exception("Failed to answer question")
        st.error(GENERIC_ERROR)
        return None


def ask_page():
    knowledge_base = activate_current_knowledge_base()
    if not knowledge_base:
        setup_screen()
        return

    render_knowledge_base_panel(knowledge_base)
    st.title(f"Ask about {knowledge_base['name']}")
    st.caption("Answers come only from these documents, and each one lists its sources.")

    problem = missing_key_message()
    if problem:
        st.warning(problem)

    with st.sidebar:
        keys = list(PIPELINES)
        pipeline_key = st.selectbox(
            "Retrieval pipeline",
            keys,
            index=keys.index(CONFIG["default_pipeline"]),
            format_func=lambda key: PIPELINES[key]["name"]
        )
        st.caption(PIPELINES[pipeline_key]["summary"])
        if st.button("Clear conversation", icon=":material/delete:", width="stretch"):
            st.session_state.messages = []
            st.rerun()
        st.caption(f"{questions_left()} questions left in this session")

    messages = st.session_state.setdefault("messages", [])

    for message in messages:
        with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
            st.markdown(message["content"])
            if message.get("footer"):
                st.caption(message["footer"])
            render_sources(message.get("sources", []))

    if not messages:
        st.markdown("##### Your documents are ready" if knowledge_base["workspace"] else "##### Try a question")
        if not knowledge_base["used_context"]:
            st.caption("This upload is large, so the extra AI context step was skipped to keep processing fast.")
        for example in knowledge_base["examples"]:
            if st.button(example, width="stretch", disabled=bool(problem)):
                st.session_state.pending_question = example
                st.rerun()

    typed = st.chat_input(
        f"Ask about {knowledge_base['name']}...",
        disabled=bool(problem),
        max_chars=CONFIG["max_question_chars"]
    )
    question = typed or st.session_state.pop("pending_question", None)
    if not question:
        return

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    refusal = validate_question(question) or spend_questions(1)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        if refusal:
            st.info(refusal)
            return
        reply = render_answer(question, pipeline_key)

    if reply:
        messages.append({"role": "user", "content": question})
        messages.append(reply)
        st.rerun()


def compare_page():
    st.title("Compare pipelines")
    knowledge_base = activate_current_knowledge_base()
    if not knowledge_base:
        st.info("Add your documents first, then come back to compare how each pipeline answers.")
        st.page_link(ask_entry, label="Add documents", icon=":material/upload_file:")
        return

    render_knowledge_base_panel(knowledge_base)
    st.caption(
        f"Ask one question about {knowledge_base['name']} and see what each pipeline finds and answers. "
        f"This uses {len(PIPELINES)} questions from your session limit."
    )

    problem = missing_key_message()
    if problem:
        st.warning(problem)

    with st.form("compare"):
        question = st.text_input(
            "Question",
            placeholder="Ask something specific from your documents",
            max_chars=CONFIG["max_question_chars"]
        )
        submitted = st.form_submit_button("Compare", type="primary", disabled=bool(problem))

    if submitted and question.strip():
        refusal = validate_question(question) or spend_questions(len(PIPELINES))
        if refusal:
            st.info(refusal)
        else:
            results = {}
            with st.spinner("Running every pipeline..."):
                for key, pipeline in PIPELINES.items():
                    try:
                        started = time.perf_counter()
                        sources = pipeline["retrieve"](question)
                        seconds = time.perf_counter() - started
                        answer = generate_answer(question, sources)
                        results[key] = {
                            "answer": link_citations(answer, sources),
                            "sources": [] if is_declined(answer) else sources,
                            "seconds": seconds
                        }
                    except Exception:
                        logger.exception("Compare failed for %s", key)
                        results[key] = {"error": GENERIC_ERROR}
            st.session_state.comparison = results

    results = st.session_state.get("comparison")
    if results:
        for tab, (key, result) in zip(st.tabs([PIPELINES[key]["name"] for key in results]), results.items()):
            with tab:
                if "error" in result:
                    st.error(result["error"])
                    continue
                st.markdown(result["answer"])
                st.caption(answer_footer(PIPELINES[key]["name"], len(result["sources"]), result["seconds"]))
                render_sources(result["sources"])


def evaluation_page():
    st.title("Evaluation")
    st.caption(
        "How often each pipeline finds the right page, tested on the FastAPI docs. "
        "Results on your own documents may differ, so use Compare to check them."
    )

    if not os.path.exists(CONFIG["eval_results_path"]):
        st.info("No results yet. Run `python eval/run_eval.py` to generate them.")
        return

    with open(CONFIG["eval_results_path"], "r", encoding="utf-8") as f:
        results = json.load(f)

    k = results["top_k"]
    rows = [
        {
            "Pipeline": row["label"],
            f"Hit rate @{k}": row["hit_rate_at_k"],
            "Hit rate @1": row["hit_rate_at_1"],
            "MRR": row["mrr"],
            "Seconds per query": row["avg_seconds"]
        }
        for row in results["pipelines"]
    ]
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            f"Hit rate @{k}": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Hit rate @1": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "MRR": st.column_config.NumberColumn(format="%.3f"),
            "Seconds per query": st.column_config.NumberColumn(format="%.2f")
        }
    )

    refusals = results.get("refusals")
    if refusals:
        st.markdown(
            f"**Out-of-scope questions:** declined {refusals['correct']} of {refusals['total']} "
            f"using the {refusals['pipeline']} pipeline."
        )

    st.markdown(
        f"""
        ##### How to read this
        - **Hit rate @{k}:** the share of questions where a correct page was among the {k} sources given to the model.
        - **Hit rate @1:** the share of questions where the very first source was from a correct page.
        - **MRR:** rewards ranking a correct page higher. 1.0 means it always came first.
        - **Seconds per query:** time to find the sources, before the model starts writing the answer.

        ##### Limits of this test
        Based on {results['question_count']} hand-written questions, so small differences may not be meaningful.
        It measures whether the right page was found, not whether the generated answer was correct.
        """
    )


with st.sidebar:
    st.markdown(f'<div class="brand">{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Learn any framework from its own docs</div>', unsafe_allow_html=True)

ask_entry = st.Page(ask_page, title="Ask", icon=":material/chat:", url_path="ask", default=True)
current_page = st.navigation(
    [
        ask_entry,
        st.Page(compare_page, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
        st.Page(evaluation_page, title="Evaluation", icon=":material/monitoring:", url_path="evaluation")
    ],
    position="top"
)
current_page.run()

with st.sidebar:
    st.divider()
    active = st.session_state.get("knowledge_base")
    if active and not active["workspace"]:
        st.markdown(f"Source: [FastAPI documentation]({SAMPLE_DOCS_URL})")
    st.caption("Answers can contain mistakes. Check the sources.")
