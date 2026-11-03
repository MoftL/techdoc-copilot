"""Offline tests. Nothing here calls OpenAI."""
from ingestion.chunkers import (
    chunk_by_sections,
    chunk_semantic,
    chunk_sliding_window,
    count_tokens,
    split_blocks,
    split_sections,
)
from ingestion.loader import clean_markdown, is_english, is_excluded, parse_pages
from retrieval.answer import NOT_FOUND_MESSAGE, build_user_prompt, generate_answer, is_declined, link_citations
from retrieval.hybrid_search import reciprocal_rank_fusion, tokenize

CODE = '```\nfrom fastapi.responses import JSONResponse\n\napp = FastAPI()\nprint(1.5)\n```'
PAGE = f"# Title\n\n## First\n\nSome intro text about responses.\n\n{CODE}\n\n## Second\n\n" + "Word " * 80


def balanced(text: str) -> bool:
    return text.count("```") % 2 == 0


def test_language_filter():
    assert is_english("https://fastapi.tiangolo.com/tutorial/cors/")
    assert is_english("https://fastapi.tiangolo.com/")
    assert not is_english("https://fastapi.tiangolo.com/tr/tutorial/cors/")
    assert not is_english("https://fastapi.tiangolo.com/zh-hant/tutorial/cors/")
    assert is_excluded("https://fastapi.tiangolo.com/release-notes/")
    assert not is_excluded("https://fastapi.tiangolo.com/reference/fastapi/")


def test_parse_pages_splits_crawl_and_handles_plain_documents():
    crawl = "----\nurl: https://x.dev/a/\n----\n\n# A\n\nBody a\n\n----\n----\nurl: https://x.dev/b/\n----\n\n# B\n\nBody b\n"
    pages = parse_pages(crawl)
    assert [(p["url"], p["title"]) for p in pages] == [("https://x.dev/a/", "A"), ("https://x.dev/b/", "B")]
    assert "----" not in pages[0]["text"]
    assert parse_pages("just text", "notes.md")[0] == {"url": "", "title": "notes.md", "text": "just text"}


def test_clean_markdown_removes_furniture_but_not_code():
    raw = (
        '[Skip to content](https://x.dev/#a)\n\n# Head[¶](https://x.dev/#a "Permanent link")\n\n'
        'See [the guide](https://x.dev/guide/) now.\n\n[Python 3.10+](#__tabbed_1_1)\n\n'
        '```\nvalue = items[0](call)\n```'
    )
    cleaned = clean_markdown(raw)
    assert cleaned == "# Head\n\nSee the guide now.\n\n```\nvalue = items[0](call)\n```"


def test_code_blocks_stay_whole_including_blank_lines():
    blocks = split_blocks(PAGE)
    assert CODE in blocks


def test_longer_fence_can_contain_a_shorter_one():
    block = "`````\ndef f():\n    \"\"\"\n    ```python\n    example()\n    ```\n    \"\"\"\n`````"
    assert split_blocks(f"Intro.\n\n{block}\n\nOutro.") == ["Intro.", block, "Outro."]
    pieces = chunk_sliding_window("`````\n" + "\n".join(f"x_{i} = {i}" for i in range(200)) + "\n`````", 50, 0)
    assert all(p.startswith("`````") and p.endswith("`````") for p in pieces)


def test_sliding_window_respects_limit_and_never_breaks_code():
    long_code = "```\n" + "\n".join(f"line_{i} = {i}" for i in range(400)) + "\n```"
    text = "\n\n".join([f"Paragraph {i} about things. " * 8 for i in range(40)] + [long_code])
    chunks = chunk_sliding_window(text, 100, 20)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)
    assert all(balanced(c) for c in chunks)


def test_oversized_single_line_is_still_split():
    chunks = chunk_sliding_window("word " * 2000, 100, 0)
    assert all(count_tokens(c) <= 100 for c in chunks)


def test_overlap_repeats_whole_blocks_only():
    paragraphs = [f"Paragraph number {i} has a few words in it." for i in range(30)]
    chunks = chunk_sliding_window("\n\n".join(paragraphs), 60, 15)
    assert chunks[1].split("\n\n")[0] in paragraphs
    assert chunks[1].split("\n\n")[0] in chunks[0]


def test_sections_track_heading_path():
    sections = split_sections(PAGE)
    names = [s["section"] for s in sections]
    assert any(name.endswith("Title > Second") for name in names)
    assert all(not s["blocks"] == [] for s in sections)


def test_section_chunks_keep_section_metadata():
    chunks = chunk_by_sections(PAGE, 800)
    assert all(c["section"] for c in chunks)
    assert all(balanced(c["text"]) for c in chunks)


def test_semantic_chunks_preserve_dotted_names_and_avoid_fragments():
    def same_topic(texts):
        return [[1.0, 0.0] for _ in texts]

    chunks = chunk_semantic(PAGE, 0.5, 400, 20, embed_fn=same_topic)
    joined = "\n".join(c["text"] for c in chunks)
    assert "from fastapi.responses import JSONResponse" in joined
    assert "print(1.5)" in joined
    assert all(balanced(c["text"]) for c in chunks)


def test_semantic_splits_on_topic_change_only_after_min_tokens():
    text = "# T\n\n" + "\n\n".join(f"Sentence {i} has several words in it here." for i in range(12))

    def alternating(texts):
        return [[1.0, 0.0] if i % 2 else [0.0, 1.0] for i in range(len(texts))]

    assert len(chunk_semantic(text, 0.5, 400, 10_000, embed_fn=alternating)) == 1
    assert len(chunk_semantic(text, 0.5, 400, 10, embed_fn=alternating)) > 1


def test_tokenizer_strips_punctuation():
    assert tokenize("What does Depends() do?") == ["what", "does", "depends", "do"]
    assert tokenize("app.get('/items_list')") == ["app", "get", "items_list"]


def test_rrf_rewards_appearing_in_both_lists():
    assert reciprocal_rank_fusion(["a", "b", "c"], ["d", "b", "e"])[0] == "b"


def test_prompt_numbers_sources():
    prompt = build_user_prompt("Why?", [{"text": "T", "section": "S", "title": "P"}])
    assert "[1] S\nT" in prompt
    assert "<question>Why?</question>" in prompt


def test_no_sources_means_no_paid_call():
    assert generate_answer("anything", []) == NOT_FOUND_MESSAGE


def test_declined_answers_are_recognized():
    assert is_declined(NOT_FOUND_MESSAGE)
    assert not is_declined("Use a Response parameter [1].")


def test_citations_become_links_outside_code_only():
    sources = [{"url": "https://x.dev/a/"}, {"url": ""}]
    linked = link_citations("Use it [1]. Also [2] and [9].\n```\nitems[1]\n```\n`a[1]`", sources)
    assert "[\\[1\\]](https://x.dev/a/)" in linked
    assert "Also [2] and [9]." in linked
    assert "items[1]" in linked and "`a[1]`" in linked
