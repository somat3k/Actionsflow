"""
Embedded Redis controller for high-performance cache management.

Uses ``fakeredis`` as an in-process Redis server so no external daemon is
required.  When ``REDIS_URL`` is set the controller transparently connects to
that server instead, giving a smooth path from development (embedded) to
production (standalone Redis).

Hierarchy
---------
RedisController
    ├── get(key) -> Optional[str]
    ├── set(key, value, ttl_seconds=None)
    ├── delete(key)
    ├── exists(key) -> bool
    ├── flush()           # flush all keys in the controller's namespace
    ├── keys(pattern)     # list keys matching glob pattern
    ├── ping() -> bool
    └── close()

All keys are automatically namespaced with ``namespace:`` so that multiple
controller instances can coexist inside the same Redis instance without
collisions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, List, Optional

log = logging.getLogger(__name__)

# ── Optional Redis client ──────────────────────────────────────────────────────
try:
    import redis as _redis_lib  # type: ignore

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REDIS_AVAILABLE = False

try:
    import fakeredis  # type: ignore

    _FAKEREDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FAKEREDIS_AVAILABLE = False


class RedisController:
    """Embedded/external Redis controller with automatic fallback.

    Parameters
    ----------
    namespace:
        String prefix applied to every key (default ``"qt"``).
    url:
        Redis URL, e.g. ``"redis://localhost:6379/0"``.  When ``None`` the
        value of the ``REDIS_URL`` environment variable is used.  If neither
        is available the controller falls back to an embedded ``fakeredis``
        server.
    default_ttl:
        Seconds a cached entry lives before expiry.  ``None`` means no expiry.
    enabled:
        When ``False`` the controller is inert – all reads return ``None`` and
        all writes are no-ops.  Useful for disabling the cache layer via config.
    """

    def __init__(
        self,
        namespace: str = "qt",
        url: Optional[str] = None,
        default_ttl: Optional[int] = 3600,
        enabled: bool = True,
    ) -> None:
        self.namespace = namespace
        self.default_ttl = default_ttl
        self._client: Any = None
        self._embedded: bool = False

        if not enabled:
            log.info("RedisController: cache disabled – running in no-op mode")
            return

        resolved_url = url or os.environ.get("REDIS_URL")

        if resolved_url and _REDIS_AVAILABLE:
            try:
                client = _redis_lib.from_url(resolved_url, decode_responses=True)
                client.ping()
                self._client = client
                log.info("RedisController: connected to %s", resolved_url)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "RedisController: cannot reach Redis at %s (%s) – falling back to embedded",
                    resolved_url,
                    exc,
                )

        # Fall back to embedded fakeredis
        if _FAKEREDIS_AVAILABLE:
            self._client = fakeredis.FakeRedis(decode_responses=True)
            self._embedded = True
            log.info("RedisController: using embedded fakeredis instance")
        else:  # pragma: no cover
            self._client = None
            log.warning(
                "RedisController: neither redis nor fakeredis available – cache disabled"
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ns(self, key: str) -> str:
        """Apply namespace prefix to a raw key."""
        return f"{self.namespace}:{key}"

    def _strip_ns(self, namespaced_key: str) -> str:
        """Remove namespace prefix from a key."""
        prefix = f"{self.namespace}:"
        if namespaced_key.startswith(prefix):
            return namespaced_key[len(prefix):]
        return namespaced_key

    # ── Public API ────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return ``True`` if the Redis client is reachable."""
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False

    def get(self, key: str) -> Optional[str]:
        """Return the raw string value for *key*, or ``None`` when absent."""
        if self._client is None:
            return None
        try:
            return self._client.get(self._ns(key))
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisController.get(%s) error: %s", key, exc)
            return None

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = -1) -> bool:
        """Store *value* for *key* with an optional TTL.

        Parameters
        ----------
        ttl_seconds:
            ``-1`` (default) uses ``self.default_ttl``.  ``None`` stores with
            no expiry.  Any positive integer sets an explicit TTL.
        """
        if self._client is None:
            return False
        ttl = self.default_ttl if ttl_seconds == -1 else ttl_seconds
        try:
            if ttl is not None and ttl > 0:
                self._client.setex(self._ns(key), ttl, value)
            else:
                self._client.set(self._ns(key), value)
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisController.set(%s) error: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """Delete *key*.  Returns ``True`` when the key existed."""
        if self._client is None:
            return False
        try:
            return bool(self._client.delete(self._ns(key)))
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisController.delete(%s) error: %s", key, exc)
            return False

    def exists(self, key: str) -> bool:
        """Return ``True`` when *key* exists in Redis."""
        if self._client is None:
            return False
        try:
            return bool(self._client.exists(self._ns(key)))
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisController.exists(%s) error: %s", key, exc)
            return False

    def ttl(self, key: str) -> Optional[int]:
        """Return remaining TTL in seconds, ``None`` if no expiry, ``-2`` if missing."""
        if self._client is None:
            return -2
        try:
            result = self._client.ttl(self._ns(key))
            # Redis returns -1 = no expiry, -2 = key missing
            if result == -1:
                return None
            return result
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisController.ttl(%s) error: %s", key, exc)
            return -2

    def keys(self, pattern: str = "*") -> List[str]:
        """Return keys matching *pattern* (glob), with namespace stripped."""
        if self._client is None:
            return []
        try:
            raw: Iterable[str] = self._client.keys(self._ns(pattern))
            return [self._strip_ns(k) for k in raw]
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisController.keys(%s) error: %s", pattern, exc)
            return []

    def keys_sample(self, pattern: str = "*", limit: int = 50) -> List[str]:
        """Return up to *limit* keys matching *pattern* using SCAN when possible."""
        if self._client is None or limit <= 0:
            return []
        try:
            if hasattr(self._client, "scan_iter"):
                results: List[str] = []
                batch_size = 100
                for key in self._client.scan_iter(self._ns(pattern), count=batch_size):
                    results.append(self._strip_ns(key))
                    if len(results) >= limit:
                        break
                return results
            raw: Iterable[str] = self._client.keys(self._ns(pattern))
            return [self._strip_ns(k) for k in list(raw)[:limit]]
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisController.keys_sample(%s) error: %s", pattern, exc)
            return []

    def flush(self) -> int:
        """Delete all keys in this controller's namespace.  Returns count deleted."""
        if self._client is None:
            return 0
        try:
            all_keys = self._client.keys(self._ns("*"))
            if not all_keys:
                return 0
            return self._client.delete(*all_keys)
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisController.flush() error: %s", exc)
            return 0

    def close(self) -> None:
        """Close the Redis connection (no-op for embedded instances)."""
        if self._client is None:
            return
        if not self._embedded:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None

    @property
    def is_available(self) -> bool:
        """``True`` when an operational Redis backend is configured."""
        return self._client is not None

    @property
    def is_embedded(self) -> bool:
        """``True`` when running against the in-process fakeredis backend."""
        return self._embedded
