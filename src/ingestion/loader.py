import hashlib
import re

LANGUAGE_CODES = {"de", "fr", "pt", "es", "ru", "uk", "tr", "hi", "ko", "ja", "zh", "zh-hant"}

# Pages about the project itself, plus a reference page that repeats /reference/fastapi/ almost word for word.
EXCLUDED_PATHS = {
    "release-notes", "fastapi-people", "management", "translations", "external-links",
    "contributing", "help-fastapi", "reference/apirouter"
}

PAGE_SPLIT = re.compile(r"(?m)^url: (https?://\S+)\s*$")
# The closing fence must match the opening length, because reference pages wrap ``` inside `````.
FENCE = re.compile(r"^[ \t]*(`{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL)
PERMALINK = re.compile(r"\[¶\]\([^)]*\)")
SKIP_LINK = re.compile(r"(?m)^\[Skip to content\]\([^)]*\)[ \t]*$")
TAB_LINK = re.compile(r"(?m)^\[[^\]]+\]\(#__tabbed_[^)]*\)[ \t]*$")
MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()\s]|\([^)]*\))+(?:\s+\"[^\"]*\")?\)")
BORDER_LINE = re.compile(r"-{3,}")
TITLE = re.compile(r"(?m)^#\s+(.+)$")


def page_path(url: str) -> str:
    return "/".join(url.split("/")[3:]).strip("/")


def is_english(url: str) -> bool:
    return page_path(url).split("/")[0] not in LANGUAGE_CODES


def is_excluded(url: str) -> bool:
    return page_path(url) in EXCLUDED_PATHS


def clean_markdown(text: str) -> str:
    """Removes page furniture and link targets, leaving code blocks exactly as written."""
    def clean_prose(prose: str) -> str:
        prose = PERMALINK.sub("", prose)
        prose = SKIP_LINK.sub("", prose)
        prose = TAB_LINK.sub("", prose)
        return MD_LINK.sub(r"\1", prose)

    parts, position = [], 0
    for match in FENCE.finditer(text):
        parts.append(clean_prose(text[position:match.start()]))
        parts.append(match.group())
        position = match.end()
    parts.append(clean_prose(text[position:]))
    return re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()


def _strip_borders(body: str) -> str:
    """The crawl wraps each page in ---- lines."""
    lines = body.strip().split("\n")
    while lines and (not lines[0].strip() or BORDER_LINE.fullmatch(lines[0].strip())):
        lines.pop(0)
    while lines and (not lines[-1].strip() or BORDER_LINE.fullmatch(lines[-1].strip())):
        lines.pop()
    return "\n".join(lines)


def _make_page(url: str, body: str, fallback_title: str = "Document") -> dict:
    text = clean_markdown(_strip_borders(body))
    title = TITLE.search(text)
    return {"url": url, "title": title.group(1).strip() if title else fallback_title, "text": text}


def parse_pages(raw_text: str, fallback_title: str = "Document") -> list[dict]:
    """Splits a crawl into pages. A plain document with no url: markers becomes a single page."""
    parts = PAGE_SPLIT.split(raw_text)
    if len(parts) == 1:
        return [_make_page("", raw_text, fallback_title)]
    return [_make_page(url, body) for url, body in zip(parts[1::2], parts[2::2])]


def parse_uploads(files: list[tuple[str, str]]) -> list[dict]:
    """Takes (name, text) pairs. Every language is kept, because this is the visitor's own document."""
    pages = []
    for file_index, (name, text) in enumerate(files):
        for page_index, page in enumerate(parse_pages(text, name)):
            if page["text"]:
                pages.append({**page, "key": f"{file_index}:{page_index}"})
    return pages


def page_key(page: dict) -> str:
    """Two uploaded files may share a title, so uploads carry their own unique key."""
    return page.get("key") or page["url"] or page["title"]


def load_pages(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        pages = parse_pages(f.read())
    return [p for p in pages if p["text"] and (not p["url"] or (is_english(p["url"]) and not is_excluded(p["url"])))]


def extract_english(raw_path: str, output_path: str) -> int:
    """Writes the English pages of the full multilingual crawl to a small file that can live in the repo."""
    pages = load_pages(raw_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"----\nurl: {page['url']}\n----\n\n{page['text']}\n\n")
    return len(pages)


def chunk_id(prefix: str, url: str, index: int) -> str:
    return f"{prefix}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:10]}_{index}"
