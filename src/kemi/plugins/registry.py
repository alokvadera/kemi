"""Plugin registry — holds the active plugins of each type on a ``MemoryService``.

Each slot is a list of plugins, allowing fan-out (e.g. multiple webhook
sinks, or multiple audit sinks). The built-in defaults (added by
``configure_webhooks`` etc.) live alongside any user-supplied plugins.

The :class:`QueryCacheProvider` slot is special: at most one cache is
active at a time (caches are stateful, so multiple would race). Use
:meth:`set_query_cache` to replace, or :meth:`disable_query_cache` to clear.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kemi.plugins._version import KEMI_PROTOCOL_VERSION, parse_version

if TYPE_CHECKING:
    from kemi.plugins.protocols import AuditSink, HookSink, QueryCacheProvider, WebhookSink

logger = logging.getLogger(__name__)


@dataclass
class PluginRegistry:
    """Holds the active plugins for one :class:`MemoryService` instance.

    The four slots are:

    * :attr:`webhook_sinks` — fan-out list of :class:`WebhookSink`.
    * :attr:`audit_sinks` — fan-out list of :class:`AuditSink`.
    * :attr:`hook_sinks` — fan-out list of :class:`HookSink`.
    * :attr:`query_cache` — single optional :class:`QueryCacheProvider`.

    The :attr:`kemi_version` field records which version of the
    :mod:`kemi.plugins.protocols` contract the plugins were written
    against. The installed ``kemi`` package exposes its own target via
    :data:`kemi.plugins.KEMI_PROTOCOL_VERSION`. Use
    :meth:`verify_compatibility` to compare them.

    Mutations are plain attribute assignments; the methods on
    :class:`MemoryService` are the supported API.
    """

    webhook_sinks: list[WebhookSink] = field(default_factory=list)
    audit_sinks: list[AuditSink] = field(default_factory=list)
    hook_sinks: list[HookSink] = field(default_factory=list)
    query_cache: QueryCacheProvider | None = None
    kemi_version: str = KEMI_PROTOCOL_VERSION

    # -- introspection helpers ----------------------------------------------

    def has_webhook_sinks(self) -> bool:
        return bool(self.webhook_sinks)

    def has_audit_sinks(self) -> bool:
        return bool(self.audit_sinks)

    def has_hook_sinks(self) -> bool:
        return bool(self.hook_sinks)

    def has_query_cache(self) -> bool:
        return self.query_cache is not None

    def clear_webhook_sinks(self) -> None:
        self.webhook_sinks.clear()

    def clear_audit_sinks(self) -> None:
        self.audit_sinks.clear()

    def clear_hook_sinks(self) -> None:
        self.hook_sinks.clear()

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for diagnostics / ``get_plugins()``."""
        return {
            "webhook_sinks": len(self.webhook_sinks),
            "audit_sinks": len(self.audit_sinks),
            "hook_sinks": len(self.hook_sinks),
            "query_cache": type(self.query_cache).__name__
            if self.query_cache is not None
            else None,
            "kemi_version": self.kemi_version,
            "kemi_protocol_version": KEMI_PROTOCOL_VERSION,
        }

    def verify_compatibility(
        self,
        *,
        strict: bool = False,
    ) -> tuple[bool, str]:
        """Compare :attr:`kemi_version` against :data:`KEMI_PROTOCOL_VERSION`.

        Returns ``(compatible, message)``. When the versions match, the
        message is informational only. When they differ, the message
        describes the mismatch. By default a mismatch is also logged at
        WARNING level; pass ``strict=True`` to raise
        :class:`~kemi.exceptions.CompatibilityError` instead.

        Compatibility is checked by SemVer major-version equality: a
        plugin built for protocol ``1.x.y`` is compatible with any
        installed protocol ``1.*.*``; a plugin built for ``2.0.0`` is
        not compatible with installed ``1.x.x``. Minor/patch
        differences are accepted.
        """
        target = KEMI_PROTOCOL_VERSION
        if self.kemi_version == target:
            return (True, f"plugin protocol {self.kemi_version} matches installed {target}")

        plugin_v = parse_version(self.kemi_version)
        target_v = parse_version(target)
        if plugin_v == (0, 0, 0) or target_v == (0, 0, 0):
            message = (
                f"plugin protocol {self.kemi_version!r} could not be compared to "
                f"installed {target!r}; one of them is not a valid MAJOR.MINOR.PATCH version"
            )
        elif plugin_v[0] != target_v[0]:
            message = (
                f"plugin protocol major version mismatch: plugin={self.kemi_version}, "
                f"installed={target} (major versions differ; plugin may not work)"
            )
        else:
            message = (
                f"plugin protocol minor/patch version difference: "
                f"plugin={self.kemi_version}, installed={target} (major version matches; "
                f"plugin should work but consider upgrading)"
            )

        if strict:
            from kemi.exceptions import CompatibilityError

            raise CompatibilityError(message)

        logger.warning(message)
        return (False, message)


__all__ = ["PluginRegistry"]
