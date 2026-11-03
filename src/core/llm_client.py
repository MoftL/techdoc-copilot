import os
from collections.abc import Iterator

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from config import CONFIG

load_dotenv(find_dotenv())

EMBEDDING_BATCH_SIZE = 100

_client = None


def has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def get_client() -> OpenAI:
    """Created on first use so the UI can start and explain a missing key instead of crashing."""
    global _client
    if _client is None:
        _client = OpenAI(max_retries=5, timeout=60.0)
    return _client


def _messages(system_prompt: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def generate_text(system_prompt: str, user_prompt: str, model: str = CONFIG["chat_model"]) -> str:
    response = get_client().chat.completions.create(
        model=model,
        messages=_messages(system_prompt, user_prompt),
        temperature=0.0
    )
    return response.choices[0].message.content


def stream_text(system_prompt: str, user_prompt: str, model: str = CONFIG["chat_model"]) -> Iterator[str]:
    stream = get_client().chat.completions.create(
        model=model,
        messages=_messages(system_prompt, user_prompt),
        temperature=0.0,
        stream=True
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def get_embeddings(texts: list[str], model: str = CONFIG["embedding_model"]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        response = get_client().embeddings.create(input=batch, model=model)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def get_embedding(text: str, model: str = CONFIG["embedding_model"]) -> list[float]:
    return get_embeddings([text], model=model)[0]
