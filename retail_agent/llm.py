"""Model providers: chat models and embeddings behind one small factory.

The rest of the code only sees LangChain's ``BaseChatModel`` and ``Embeddings`` interfaces,
so OpenAI, Gemini or a test double can be swapped without touching the agent.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from retail_agent.config import Settings
from retail_agent.rag.hashing import hash_embed


class HashingEmbeddings(Embeddings):
    """Deterministic, offline embeddings using the hashing trick.

    Each text becomes a sparse bag of unigrams and bigrams, hashed into ``dim`` buckets with a
    sign bit (to reduce collision bias), weighted by 1 + log(tf) and L2-normalised so that the
    inner product equals cosine similarity. No model download and no API key, which keeps
    tests and the default demo free. It captures lexical overlap, not meaning; use OpenAI or
    Gemini embeddings for semantic search.

    The maths lives in ``retail_agent.rag.hashing`` (NumPy only) so worker processes start fast.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed_one(self, text: str) -> list[float]:
        return hash_embed(text, self.dim).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_one(text)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "local":
        return HashingEmbeddings(dim=settings.local_embedding_dim)
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model)
    if settings.embedding_provider == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("Install the gemini extra: pip install -e '.[gemini]'") from exc
        return GoogleGenerativeAIEmbeddings(model=settings.gemini_embedding_model)
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def embedding_signature(settings: Settings) -> str:
    """Identifies the embedding space; an index built with another signature must be rebuilt."""
    if settings.embedding_provider == "local":
        return f"local-hashing-{settings.local_embedding_dim}"
    if settings.embedding_provider == "openai":
        return f"openai-{settings.openai_embedding_model}"
    return f"gemini-{settings.gemini_embedding_model}"


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    if settings.llm_provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("Install the gemini extra: pip install -e '.[gemini]'") from exc
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def model_signature(settings: Settings) -> str:
    return settings.openai_model if settings.llm_provider == "openai" else settings.gemini_model
