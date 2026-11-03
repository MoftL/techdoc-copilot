import re
from collections.abc import Callable

import numpy as np
import tiktoken

from core.llm_client import get_embeddings
from ingestion.loader import FENCE

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*\[(])")
HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
SMALL_SECTION_TOKENS = 40

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text, disallowed_special=()))


def _join(blocks: list[str]) -> str:
    return "\n\n".join(blocks)


def _split_prose(text: str) -> list[str]:
    blocks = []
    for paragraph in re.split(r"\n\s*\n", text):
        current = []
        for line in paragraph.strip().split("\n"):
            if HEADING.match(line):
                if current:
                    blocks.append("\n".join(current))
                    current = []
                blocks.append(line)
            elif line.strip():
                current.append(line)
        if current:
            blocks.append("\n".join(current))
    return blocks


def split_blocks(text: str) -> list[str]:
    """Paragraphs, headings and whole code blocks. These are the units no chunker may cut through."""
    blocks, position = [], 0
    for match in FENCE.finditer(text):
        blocks.extend(_split_prose(text[position:match.start()]))
        blocks.append(match.group().strip())
        position = match.end()
    blocks.extend(_split_prose(text[position:]))
    return blocks


def _hard_split(text: str, max_tokens: int) -> list[str]:
    tokens = _encoding.encode(text, disallowed_special=())
    return [_encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def _pack(parts: list[str], max_tokens: int, separator: str) -> list[str]:
    packed, current = [], ""
    for part in parts:
        candidate = f"{current}{separator}{part}" if current else part
        if count_tokens(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            packed.append(current)
        if count_tokens(part) <= max_tokens:
            current = part
        else:
            pieces = _hard_split(part, max_tokens)
            packed.extend(pieces[:-1])
            current = pieces[-1]
    if current:
        packed.append(current)
    return packed


def _fit_block(block: str, max_tokens: int) -> list[str]:
    if count_tokens(block) <= max_tokens:
        return [block]

    if block.startswith("```"):
        # Each piece of a long code block is re-wrapped so it still renders as code.
        lines = block.split("\n")
        opening = lines[0]
        closing = opening[:len(opening) - len(opening.lstrip("`"))]
        room = max(1, max_tokens - count_tokens(f"{opening}\n\n{closing}") - 2)
        return [f"{opening}\n{piece}\n{closing}" for piece in _pack(lines[1:-1], room, "\n")]

    sentences = SENTENCE_END.split(block)
    if len(sentences) > 1:
        return _pack(sentences, max_tokens, " ")
    return _pack(block.split("\n"), max_tokens, "\n")


def fit_blocks(blocks: list[str], max_tokens: int) -> list[str]:
    return [piece for block in blocks for piece in _fit_block(block, max_tokens)]


def _overlap_tail(blocks: list[str], overlap_size: int, room: int) -> list[str]:
    """Repeats whole trailing blocks, so an overlap never starts in the middle of a word or code line."""
    tail = []
    for block in reversed(blocks):
        if count_tokens(_join([block] + tail)) > min(overlap_size, room):
            break
        tail.insert(0, block)
    return tail


def _window(blocks: list[str], target_chunk_size: int, overlap_size: int) -> list[str]:
    chunks, current = [], []
    for block in fit_blocks(blocks, target_chunk_size):
        if current and count_tokens(_join(current + [block])) > target_chunk_size:
            chunks.append(_join(current))
            room = target_chunk_size - count_tokens(block) - 2
            current = _overlap_tail(current, overlap_size, room) if overlap_size and room > 0 else []
            if current and count_tokens(_join(current + [block])) > target_chunk_size:
                current = []
        current.append(block)
    if current:
        chunks.append(_join(current))
    return chunks


def chunk_sliding_window(text: str, target_chunk_size: int = 400, overlap_size: int = 50) -> list[str]:
    return _window(split_blocks(text), target_chunk_size, overlap_size)


def split_sections(text: str) -> list[dict]:
    """Groups blocks under their headings. Near-empty sections join the next one."""
    sections, stack = [], []
    current = {"section": "", "blocks": []}

    for block in split_blocks(text):
        heading = HEADING.match(block)
        if heading:
            if current["blocks"]:
                sections.append(current)
            level = len(heading.group(1))
            stack = [item for item in stack if item[0] < level] + [(level, heading.group(2).strip())]
            current = {"section": " > ".join(name for _, name in stack), "blocks": [block]}
        else:
            current["blocks"].append(block)
    if current["blocks"]:
        sections.append(current)

    merged, carry = [], []
    for section in sections:
        blocks = carry + section["blocks"]
        if count_tokens(_join(blocks)) < SMALL_SECTION_TOKENS:
            carry = blocks
        else:
            merged.append({"section": section["section"], "blocks": blocks})
            carry = []
    if carry:
        if merged:
            merged[-1]["blocks"].extend(carry)
        else:
            merged.append({"section": sections[-1]["section"], "blocks": carry})
    return merged


def chunk_by_sections(text: str, max_tokens: int = 800) -> list[dict]:
    chunks = []
    for section in split_sections(text):
        for chunk in _window(section["blocks"], max_tokens, 0):
            chunks.append({"text": chunk, "section": section["section"]})
    return chunks


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_prod = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return float(dot_prod / (norm_a * norm_b))


def _semantic_units(blocks: list[str], max_tokens: int) -> list[str]:
    """A code block stays with the sentence that introduces it, because each means little alone."""
    units = []
    for block in fit_blocks(blocks, max_tokens):
        if block.startswith("```") and units and count_tokens(f"{units[-1]}\n\n{block}") <= max_tokens:
            units[-1] = f"{units[-1]}\n\n{block}"
        else:
            units.append(block)
    return units


def chunk_semantic(
    text: str,
    similarity_threshold: float = 0.5,
    max_tokens: int = 400,
    min_tokens: int = 120,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None
) -> list[dict]:
    """
    Starts a new chunk at a heading, at the size limit, or where the topic changes.
    A topic change only counts once the chunk has reached min_tokens, which prevents tiny fragments.
    """
    sections = [
        {"section": s["section"], "units": _semantic_units(s["blocks"], max_tokens)}
        for s in split_sections(text)
    ]
    all_units = [unit for s in sections for unit in s["units"]]
    if not all_units:
        return []

    vectors = iter((embed_fn or get_embeddings)(all_units))
    chunks = []
    for section in sections:
        current, previous_vector = [], None
        for unit in section["units"]:
            vector = next(vectors)
            if current:
                too_big = count_tokens(_join(current + [unit])) > max_tokens
                big_enough = count_tokens(_join(current)) >= min_tokens
                topic_changed = cosine_similarity(previous_vector, vector) < similarity_threshold
                if too_big or (big_enough and topic_changed):
                    chunks.append({"text": _join(current), "section": section["section"]})
                    current = []
            current.append(unit)
            previous_vector = vector
        if current:
            chunks.append({"text": _join(current), "section": section["section"]})
    return chunks
