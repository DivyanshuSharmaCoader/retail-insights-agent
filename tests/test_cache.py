from __future__ import annotations

import time

import fakeredis.aioredis

from retail_agent.cache import AnswerCache, TTLCache


def test_ttl_expiry_and_lru_eviction() -> None:
    cache = TTLCache(maxsize=2, ttl_s=0.05)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # 'a' becomes most recently used
    cache.set("c", 3)  # evicts 'b'
    assert cache.get("b") is None and cache.get("a") == 1
    time.sleep(0.08)
    assert cache.get("a") is None


def test_question_normalisation_shares_keys() -> None:
    c = AnswerCache()
    assert c.key_for("Total revenue in Q2?", "m") == c.key_for("  total   revenue in q2 ", "m")
    assert c.key_for("Total revenue in Q2?", "m") != c.key_for("Total revenue in Q2?", "other")


async def test_redis_tier_is_shared_between_instances() -> None:
    server = fakeredis.FakeServer()
    first = AnswerCache(fakeredis.aioredis.FakeRedis(server=server))
    await first.set("k", {"answer": "42"})
    second = AnswerCache(fakeredis.aioredis.FakeRedis(server=server))  # empty memory tier
    assert await second.get("k") == {"answer": "42"}
    assert second.stats["redis_hits"] == 1
    assert await second.get("k") == {"answer": "42"}  # now served from memory
    assert second.stats["memory_hits"] == 1


class BrokenRedis:
    def __init__(self) -> None:
        self.calls = 0

    async def get(self, key: str) -> None:
        self.calls += 1
        raise ConnectionError("redis is down")

    async def set(self, *args, **kwargs) -> None:
        self.calls += 1
        raise ConnectionError("redis is down")


async def test_redis_outage_degrades_to_memory() -> None:
    broken = BrokenRedis()
    cache = AnswerCache(broken, cooldown_s=60)
    assert await cache.get("k") is None  # error is swallowed, breaker opens
    await cache.set("k", {"answer": "ok"})  # memory only; Redis skipped during cooldown
    assert await cache.get("k") == {"answer": "ok"}
    assert broken.calls == 1 and cache.stats["redis_errors"] == 1
