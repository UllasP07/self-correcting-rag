"""Cloud provider: Azure OpenAI (chat via GPT-4o mini, embeddings via
text-embedding-3-small).

Mirrors ollama_client's embed()/chat() exactly, so nothing downstream changes.
Selected by CHAT_PROVIDER / EMBED_PROVIDER = "azure" in config.

KEY AZURE QUIRK: the `model` you pass is your *deployment name* (chosen in the
Azure OpenAI portal when you deploy a model), not the underlying model name.
Auth is endpoint + API key (from the portal). The `openai` SDK's AzureOpenAI
client speaks the same surface as the regular OpenAI client.
"""
from __future__ import annotations

from typing import Iterable

from .config import settings

_client = None


def _azure():
    """Lazily build the client so importing this module never requires creds."""
    global _client
    if _client is None:
        from openai import AzureOpenAI

        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
            raise RuntimeError(
                "Azure provider selected but AZURE_OPENAI_ENDPOINT / "
                "AZURE_OPENAI_API_KEY are not set in .env"
            )
        _client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    return _client


def chat(system: str, user: str, temperature: float = 0.1) -> str:
    resp = _azure().chat.completions.create(
        model=settings.azure_chat_deployment,  # deployment name
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


def embed(texts: Iterable[str]) -> list[list[float]]:
    texts = list(texts)
    if not texts:
        return []
    resp = _azure().embeddings.create(
        model=settings.azure_embed_deployment,  # deployment name
        input=texts,
    )
    # Azure returns embeddings in input order.
    return [d.embedding for d in resp.data]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
