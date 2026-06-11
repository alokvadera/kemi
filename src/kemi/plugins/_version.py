"""Plugin-protocol version constants.

The :class:`kemi.plugins.registry.PluginRegistry` carries a
``kemi_version`` field that records which version of the
:mod:`kemi.plugins.protocols` contract the plugins were written against.
The installed ``kemi`` package exposes a target version via
:data:`KEMI_PROTOCOL_VERSION`. :meth:`PluginRegistry.verify_compatibility`
compares the two.

This is a **contract** version, distinct from the package version in
``pyproject.toml``. The contract only advances when a Protocol in
``kemi.plugins.protocols`` changes shape (added/removed methods,
signature change). The package version may bump many times between
contract bumps.
"""

from __future__ import annotations

KEMI_PROTOCOL_VERSION: str = "1.0.0"


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse ``"MAJOR.MINOR.PATCH"`` into a comparable 3-tuple.

    Trailing pre-release / build-metadata suffixes are stripped. Anything
    that does not look like ``"<int>.<int>.<int>"`` after stripping falls
    back to ``(0, 0, 0)`` so a malformed value never blocks loading.
    """
    head = version.split("-", 1)[0].split("+", 1)[0]
    parts = head.split(".")
    if len(parts) != 3:
        return (0, 0, 0)
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            return (0, 0, 0)
    return (out[0], out[1], out[2])


__all__ = ["KEMI_PROTOCOL_VERSION", "parse_version"]
