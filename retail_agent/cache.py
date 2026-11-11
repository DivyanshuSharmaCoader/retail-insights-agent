"""Two-tier answer cache: in-process TTL LRU in front of Redis.

* Tier 1 (memory) answers repeated questions with no network hop, per worker process.
* Tier 2 (Redis) is shared by all workers and survives restarts.
* If Redis errors, the cache stops calling it for ``cooldown_s`` (a minimal circuit
  breaker) and keeps serving from memory, so a Redis outage never fails a request.

Only stateless, verified answers are cached. The key includes the model and index
signatures, so changing the model or rebuilding the index never serves stale answers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any

log = logging.getLogger(__name__)


class TTLCache:
    """Bounded LRU with per-entry expiry. Thread-safe (the lock also covers sync callers)."""

    def __init__(self, maxsize: int = 256, ttl_s: float = 3600) -> None:
        self.maxsize, self.ttl_s = maxsize, ttl_s
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip("?.! ")


class AnswerCache:
    def __init__(
        self,
        redis_client: Any | None = None,
        ttl_s: int = 3600,
        maxsize: int = 256,
        namespace: str = "ria:answer:v1",
        cooldown_s: float = 30.0,
    ) -> None:
        self.memory = TTLCache(maxsize, ttl_s)
        self.redis = redis_client
        self.ttl_s = ttl_s
        self.namespace = namespace
        self.cooldown_s = cooldown_s
        self._redis_down_until = 0.0
        self.stats = {"memory_hits": 0, "redis_hits": 0, "misses": 0, "redis_errors": 0}

    @classmethod
    def from_url(cls, redis_url: str | None, **kwargs: Any) -> AnswerCache:
        client = None
        if redis_url:
            import redis.asyncio as aioredis

            client = aioredis.from_url(redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)
        return cls(client, **kwargs)

    @property
    def backend(self) -> str:
        return "redis+memory" if self.redis is not None else "memory"

    def key_for(self, question: str, signature: str) -> str:
        digest = hashlib.sha256(f"{signature}|{normalise_question(question)}".encode()).hexdigest()
        return f"{self.namespace}:{digest[:32]}"

    def _redis_available(self) -> bool:
        return self.redis is not None and time.monotonic() >= self._redis_down_until

    def _trip(self, exc: Exception) -> None:
        self.stats["redis_errors"] += 1
        self._redis_down_until = time.monotonic() + self.cooldown_s
        log.warning("redis unavailable, using memory cache only",
                    extra={"fields": {"error": type(exc).__name__, "cooldown_s": self.cooldown_s}})

    async def get(self, key: str) -> dict | None:
        value = self.memory.get(key)
        if value is not None:
            self.stats["memory_hits"] += 1
            return value
        if self._redis_available():
            try:
                raw = await self.redis.get(key)
            except Exception as exc:  # any Redis/network failure degrades to memory
                self._trip(exc)
            else:
                if raw is not None:
                    value = json.loads(raw)
                    self.memory.set(key, value)
                    self.stats["redis_hits"] += 1
                    return value
        self.stats["misses"] += 1
        return None

    async def set(self, key: str, value: dict) -> None:
        self.memory.set(key, value)
        if self._redis_available():
            try:
                await self.redis.set(key, json.dumps(value, default=str), ex=self.ttl_s)
            except Exception as exc:
                self._trip(exc)

    async def close(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()
